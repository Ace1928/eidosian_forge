from typing import List, Tuple
import numpy as np
from qiskit import circuit
from qiskit.visualization.timeline import types
def qreg_creg_descending(bits: List[types.Bits]) -> List[types.Bits]:
    """Sort bits by descending order.

    Bit order becomes Q_N, Q_N-1, ..., Cl_N, Cl_N-1, ...

    Args:
        bits: List of bits to sort.

    Returns:
        Sorted bits.
    """
    return [x for x in bits[::-1] if isinstance(x, circuit.Qubit)] + [x for x in bits[::-1] if isinstance(x, circuit.Clbit)]