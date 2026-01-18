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
def has_mid_circuit_measurements(circuit: qml.tape.QuantumScript):
    """Returns ``True`` if the circuit contains a ``MidMeasureMP`` object and ``False`` otherwise.

    Args:
        circuit (QuantumTape): A ``QuantumScript``

    Returns:
        bool: Whether the circuit contains a ``MidMeasureMP`` object
    """
    return any((isinstance(op, MidMeasureMP) for op in circuit.operations))