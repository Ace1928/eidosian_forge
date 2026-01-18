from dataclasses import replace
from functools import partial
from typing import Union, Tuple, Sequence
import concurrent.futures
import numpy as np
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.typing import Result, ResultBatch
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.transforms.core import TransformProgram
from pennylane.measurements import (
from pennylane.ops.qubit.observables import BasisStateProjector
from . import Device
from .execution_config import ExecutionConfig, DefaultExecutionConfig
from .default_qubit import accepted_sample_measurement
from .modifiers import single_tape_support, simulator_tracking
from .preprocess import (
def measure_analytical(self, circuit, stim_circuit, tableau_simulator, global_phase):
    """Given a circuit, compute tableau and return the analytical measurement results."""
    measurement_map = {DensityMatrixMP: self._measure_density_matrix, StateMP: self._measure_state, ExpectationMP: self._measure_expectation, VarianceMP: self._measure_variance, VnEntropyMP: self._measure_vn_entropy, MutualInfoMP: self._measure_mutual_info, PurityMP: self._measure_purity, ProbabilityMP: self._measure_probability}
    results = []
    for meas in circuit.measurements:
        measurement_func = measurement_map.get(type(meas), None)
        if measurement_func is None:
            raise NotImplementedError(f"default.clifford doesn't support the {type(meas)} measurement analytically at the moment.")
        results.append(measurement_func(meas, tableau_simulator, circuit=circuit, stim_circuit=stim_circuit, global_phase=global_phase))
    return results