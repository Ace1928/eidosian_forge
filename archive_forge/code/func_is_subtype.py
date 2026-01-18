from __future__ import annotations
import enum
from .types import Type, Bool, Uint
def is_subtype(left: Type, right: Type, /, strict: bool=False) -> bool:
    """Does the relation :math:`\\text{left} \\le \\text{right}` hold?  If there is no ordering
    relation between the two types, then this returns ``False``.  If ``strict``, then the equality
    is also forbidden.

    Examples:
        Check if one type is a subclass of another::

            >>> from qiskit.circuit.classical import types
            >>> types.is_subtype(types.Uint(8), types.Uint(16))
            True

        Check if one type is a strict subclass of another::

            >>> types.is_subtype(types.Bool(), types.Bool())
            True
            >>> types.is_subtype(types.Bool(), types.Bool(), strict=True)
            False
    """
    order_ = order(left, right)
    return order_ is Ordering.LESS or (not strict and order_ is Ordering.EQUAL)