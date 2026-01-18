from __future__ import annotations
from typing import TypeVar
from qiskit.quantum_info.operators.linear_op import LinearOp
def anti_commutator(a: OperatorTypeT, b: OperatorTypeT) -> OperatorTypeT:
    """Compute anti-commutator of a and b.

    .. math::

        ab + ba.

    Args:
        a: Operator a.
        b: Operator b.
    Returns:
        The anti-commutator
    """
    return a @ b + b @ a