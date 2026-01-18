from __future__ import annotations
import enum
from .types import Type, Bool, Uint
def cast_kind(from_: Type, to_: Type, /) -> CastKind:
    """Determine the sort of cast that is required to move from the left type to the right type.

    Examples:

        .. code-block:: python

            >>> from qiskit.circuit.classical import types
            >>> types.cast_kind(types.Bool(), types.Bool())
            <CastKind.EQUAL: 1>
            >>> types.cast_kind(types.Uint(8), types.Bool())
            <CastKind.IMPLICIT: 2>
            >>> types.cast_kind(types.Bool(), types.Uint(8))
            <CastKind.LOSSLESS: 3>
            >>> types.cast_kind(types.Uint(16), types.Uint(8))
            <CastKind.DANGEROUS: 4>
    """
    if (coercer := _ALLOWED_CASTS.get((from_.kind, to_.kind))) is None:
        return CastKind.NONE
    return coercer(from_, to_)