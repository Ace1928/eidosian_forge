from collections import Counter
from typing import Optional, Sequence
import warnings
from numpy.random import default_rng
import numpy as np
import pennylane as qml
from pennylane.measurements import (
from pennylane.typing import Result
from .initialize_state import create_initial_state
from .apply_operation import apply_operation
from .measure import measure
from .sampling import measure_with_samples
def init_auxiliary_circuit(circuit: qml.tape.QuantumScript):
    """Creates an auxiliary circuit to perform one-shot mid-circuit measurement calculations.

    Measurements are replaced by SampleMP measurements on wires and observables found in the
    original measurements.

    Args:
        circuit (QuantumTape): The original QuantumScript

    Returns:
        QuantumScript: A copy of the circuit with modified measurements
    """
    new_measurements = []
    for m in circuit.measurements:
        if not m.mv:
            if isinstance(m, VarianceMP):
                new_measurements.append(SampleMP(obs=m.obs))
            else:
                new_measurements.append(m)
    return qml.tape.QuantumScript(circuit.operations, new_measurements, shots=1, trainable_params=circuit.trainable_params)