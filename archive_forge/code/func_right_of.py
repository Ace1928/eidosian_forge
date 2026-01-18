from typing import Dict, List, Tuple, TYPE_CHECKING
import cirq
def right_of(qubit: cirq.GridQubit) -> cirq.GridQubit:
    """Gives node with one unit more on the first coordinate.

    Args:
        qubit: Reference node.

    Returns:
        New translated node.
    """
    return cirq.GridQubit(qubit.row + 1, qubit.col)