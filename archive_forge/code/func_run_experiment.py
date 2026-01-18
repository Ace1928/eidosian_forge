import itertools
import re
import socket
import subprocess
import warnings
from contextlib import contextmanager
from math import pi, log
from typing import (
import httpx
import networkx as nx
import numpy as np
from qcs_api_client.client import QCSClientConfiguration
from qcs_api_client.models import ListQuantumProcessorsResponse
from qcs_api_client.operations.sync import list_quantum_processors
from rpcq.messages import ParameterAref
from pyquil.api import EngagementManager
from pyquil.api._abstract_compiler import AbstractCompiler, QuantumExecutable
from pyquil.api._compiler import QPUCompiler, QVMCompiler
from pyquil.api._qam import QAM, QAMExecutionResult
from pyquil.api._qcs_client import qcs_client
from pyquil.api._qpu import QPU
from pyquil.api._qvm import QVM
from pyquil.experiment._main import Experiment
from pyquil.experiment._memory import merge_memory_map_lists
from pyquil.experiment._result import ExperimentResult, bitstrings_to_expectations
from pyquil.experiment._setting import ExperimentSetting
from pyquil.external.rpcq import CompilerISA
from pyquil.gates import RX, MEASURE
from pyquil.noise import decoherence_noise_with_asymmetric_ro, NoiseModel
from pyquil.paulis import PauliTerm
from pyquil.pyqvm import PyQVM
from pyquil.quantum_processor import (
from pyquil.quil import Program
def run_experiment(self, experiment: Experiment, memory_map: Optional[Mapping[str, Sequence[Union[int, float]]]]=None) -> List[ExperimentResult]:
    """
        Run an ``Experiment`` on a QVM or QPU backend. An ``Experiment`` is composed of:

            - A main ``Program`` body (or ansatz).
            - A collection of ``ExperimentSetting`` objects, each of which encodes a particular
              state preparation and measurement.
            - A ``SymmetrizationLevel`` for enacting different readout symmetrization strategies.
            - A number of shots to collect for each (unsymmetrized) ``ExperimentSetting``.

        Because the main ``Program`` is static from run to run of an ``Experiment``, we can leverage
        our platform's Parametric Compilation feature. This means that the ``Program`` can be
        compiled only once, and the various alterations due to state preparation, measurement,
        and symmetrization can all be realized at runtime by providing a ``memory_map``. Thus, the
        steps in the ``experiment`` method are as follows:

           1. Generate a parameterized program corresponding to the ``Experiment``
              (see the ``Experiment.generate_experiment_program()`` method for more
              details on how it changes the main body program to support state preparation,
              measurement, and symmetrization).

            2. Compile the parameterized program into a parametric (binary) executable, which
               contains declared variables that can be assigned at runtime.

            3. For each ``ExperimentSetting`` in the ``Experiment``, we repeat the following:

                a. Build a collection of memory maps that correspond to the various state
                   preparation, measurement, and symmetrization specifications.
                b. Run the parametric executable on the QVM or QPU backend, providing the memory map
                   to assign variables at runtime.
                c. Extract the desired statistics from the classified bitstrings that are produced
                   by the QVM or QPU backend, and package them in an ``ExperimentResult`` object.

            3. Return the list of ``ExperimentResult`` objects.

        This method is extremely useful shorthand for running near-term applications and algorithms,
        which often have this ansatz + settings structure.

        :param experiment: The ``Experiment`` to run.
        :param memory_map: A dictionary mapping declared variables / parameters to their values.
            The values are a list of floats or integers. Each float or integer corresponds to
            a particular classical memory register. The memory map provided to the ``experiment``
            method corresponds to variables in the main body program that we would like to change
            at runtime (e.g. the variational parameters provided to the ansatz of the variational
            quantum eigensolver).
        :return: A list of ``ExperimentResult`` objects containing the statistics gathered
            according to the specifications of the ``Experiment``.
        """
    experiment_program = experiment.generate_experiment_program()
    executable = self.compile(experiment_program)
    if memory_map is None:
        memory_map = {}
    results = []
    for settings in experiment:
        if len(settings) > 1:
            raise ValueError('We only support length-1 settings for now.')
        setting = settings[0]
        qubits = cast(List[int], setting.out_operator.get_qubits())
        experiment_setting_memory_map = experiment.build_setting_memory_map(setting)
        symmetrization_memory_maps = experiment.build_symmetrization_memory_maps(qubits)
        merged_memory_maps = merge_memory_map_lists([experiment_setting_memory_map], symmetrization_memory_maps)
        all_bitstrings = []
        for merged_memory_map in merged_memory_maps:
            final_memory_map = {**memory_map, **merged_memory_map}
            executable_copy = executable.copy()
            final_memory_map = cast(Mapping[Union[str, ParameterAref], Union[int, float]], final_memory_map)
            executable_copy._memory.write(final_memory_map)
            bitstrings = self.run(executable_copy).readout_data.get('ro')
            assert bitstrings is not None
            if 'symmetrization' in final_memory_map:
                bitmask = np.array(np.array(final_memory_map['symmetrization']) / np.pi, dtype=int)
                bitstrings = np.bitwise_xor(bitstrings, bitmask)
            all_bitstrings.append(bitstrings)
        symmetrized_bitstrings = np.concatenate(all_bitstrings)
        joint_expectations = [experiment.get_meas_registers(qubits)]
        if setting.additional_expectations:
            joint_expectations += setting.additional_expectations
        expectations = bitstrings_to_expectations(symmetrized_bitstrings, joint_expectations=joint_expectations)
        means = np.mean(expectations, axis=0)
        std_errs = np.std(expectations, axis=0, ddof=1) / np.sqrt(len(expectations))
        joint_results = []
        for qubit_subset, mean, std_err in zip(joint_expectations, means, std_errs):
            out_operator = PauliTerm.from_list([(setting.out_operator[i], i) for i in qubit_subset])
            s = ExperimentSetting(in_state=setting.in_state, out_operator=out_operator, additional_expectations=None)
            r = ExperimentResult(setting=s, expectation=mean, std_err=std_err, total_counts=len(expectations))
            joint_results.append(r)
        result = ExperimentResult(setting=setting, expectation=joint_results[0].expectation, std_err=joint_results[0].std_err, total_counts=joint_results[0].total_counts, additional_results=joint_results[1:])
        results.append(result)
    return results