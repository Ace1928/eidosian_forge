import itertools
import warnings
from math import log, pi
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union, cast
import networkx as nx
import numpy as np
from qcs_api_client.client import QCSClientConfiguration
from pyquil.api._compiler import AbstractCompiler, QVMCompiler
from pyquil.api._qam import QAM
from pyquil.api._qpu import QPU
from pyquil.api._quantum_computer import QuantumComputer as QuantumComputerV3
from pyquil.api._quantum_computer import get_qc as get_qc_v3, QuantumExecutable
from pyquil.api._qvm import QVM
from pyquil.experiment._main import Experiment
from pyquil.experiment._memory import merge_memory_map_lists
from pyquil.experiment._result import ExperimentResult, bitstrings_to_expectations
from pyquil.experiment._setting import ExperimentSetting
from pyquil.gates import MEASURE, RX
from pyquil.noise import NoiseModel, decoherence_noise_with_asymmetric_ro
from pyquil.paulis import PauliTerm
from pyquil.pyqvm import PyQVM
from pyquil.quantum_processor import AbstractQuantumProcessor, NxQuantumProcessor
from pyquil.quil import Program, validate_supported_quil
from pyquil.quilatom import qubit_index
from ._qam import StatefulQAM
def run_and_measure(self, program: Program, trials: int) -> Dict[int, np.ndarray]:
    """
        Run the provided state preparation program and measure all qubits.

        The returned data is a dictionary keyed by qubit index because qubits for a given
        QuantumComputer may be non-contiguous and non-zero-indexed. To turn this dictionary
        into a 2d numpy array of bitstrings, consider::

            bitstrings = qc.run_and_measure(...)
            bitstring_array = np.vstack([bitstrings[q] for q in qc.qubits()]).T
            bitstring_array.shape  # (trials, len(qc.qubits()))

        .. note::

            If the target :py:class:`QuantumComputer` is a noiseless :py:class:`QVM` then
            only the qubits explicitly used in the program will be measured. Otherwise all
            qubits will be measured. In some circumstances this can exhaust the memory
            available to the simulator, and this may be manifested by the QVM failing to
            respond or timeout.

        .. note::

            In contrast to :py:class:`QVMConnection.run_and_measure`, this method simulates
            noise correctly for noisy QVMs. However, this method is slower for ``trials > 1``.
            For faster noise-free simulation, consider
            :py:class:`WavefunctionSimulator.run_and_measure`.

        :param program: The state preparation program to run and then measure.
        :param trials: The number of times to run the program.
        :return: A dictionary keyed by qubit index where the corresponding value is a 1D array of
            measured bits.
        """
    program = program.copy()
    validate_supported_quil(program)
    ro = program.declare('ro', 'BIT', len(self.qubits()))
    measure_used = isinstance(self.qam, QVM) and self.qam.noise_model is None
    qubits_to_measure = set(map(qubit_index, program.get_qubits()) if measure_used else self.qubits())
    for i, q in enumerate(qubits_to_measure):
        program.inst(MEASURE(q, ro[i]))
    program.wrap_in_numshots_loop(trials)
    executable = self.compile(program)
    bitstring_array = self.run(executable=executable)
    bitstring_dict = {}
    for i, q in enumerate(qubits_to_measure):
        bitstring_dict[q] = bitstring_array[:, i]
    for q in set(self.qubits()) - set(qubits_to_measure):
        bitstring_dict[q] = np.zeros(trials)
    return bitstring_dict