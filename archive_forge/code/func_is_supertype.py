from __future__ import annotations
import enum
from .types import Type, Bool, Uint
def is_supertype(left: Type, right: Type, /, strict: bool=False) -> bool:
    """Does the relation :math:`\\text{left} \\ge \\text{right}` hold?  If there is no ordering
    relation between the two types, then this returns ``False``.  If ``strict``, then the equality
    is also forbidden.

    Examples:
        Check if one type is a superclass of another::

            >>> from qiskit.circuit.classical import types
            >>> types.is_supertype(types.Uint(8), types.Uint(16))
            False

        Check if one type is a strict superclass of another::

            >>> types.is_supertype(types.Bool(), types.Bool())
            True
            >>> types.is_supertype(types.Bool(), types.Bool(), strict=True)
            False
    """
    order_ = order(left, right)
    return order_ is Ordering.GREATER or (not strict and order_ is Ordering.EQUAL)