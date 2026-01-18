from typing import List, Tuple
import numpy as np
from qiskit import circuit
from qiskit.visualization.timeline import types
def qreg_creg_ascending(bits: List[types.Bits]) -> List[types.Bits]:
    """Sort bits by ascending order.

    Bit order becomes Q0, Q1, ..., Cl0, Cl1, ...

    Args:
        bits: List of bits to sort.

    Returns:
        Sorted bits.
    """
    return [x for x in bits if isinstance(x, circuit.Qubit)] + [x for x in bits if isinstance(x, circuit.Clbit)]