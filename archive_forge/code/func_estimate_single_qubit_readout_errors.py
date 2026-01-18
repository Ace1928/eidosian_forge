import dataclasses
import time
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING
import sympy
import numpy as np
from cirq import circuits, ops, study
def estimate_single_qubit_readout_errors(sampler: 'cirq.Sampler', *, qubits: Iterable['cirq.Qid'], repetitions: int=1000) -> SingleQubitReadoutCalibrationResult:
    """Estimate single-qubit readout error.

    For each qubit, prepare the |0⟩ state and measure. Calculate how often a 1
    is measured. Also, prepare the |1⟩ state and calculate how often a 0 is
    measured. The state preparations and measurements are done in parallel,
    i.e., for the first experiment, we actually prepare every qubit in the |0⟩
    state and measure them simultaneously.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        qubits: The qubits being tested.
        repetitions: The number of measurement repetitions to perform.

    Returns:
        A SingleQubitReadoutCalibrationResult storing the readout error
        probabilities as well as the number of repetitions used to estimate
        the probabilities. Also stores a timestamp indicating the time when
        data was finished being collected from the sampler.
    """
    num_qubits = len(list(qubits))
    return estimate_parallel_single_qubit_readout_errors(sampler=sampler, qubits=qubits, repetitions=repetitions, trials=2, bit_strings=np.array([[0] * num_qubits, [1] * num_qubits]))