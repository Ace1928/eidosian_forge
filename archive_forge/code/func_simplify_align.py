from __future__ import annotations
import dataclasses
import enum
import typing
def simplify_align(align_type: Literal['left', 'center', 'right', 'relative', WHSettings.RELATIVE] | Align, align_amount: int | None) -> Align | tuple[Literal[WHSettings.RELATIVE], int]:
    """
    Recombine (align_type, align_amount) into an align value.
    Inverse of normalize_align.
    """
    if align_type == WHSettings.RELATIVE:
        if not isinstance(align_amount, int):
            raise TypeError(align_amount)
        return (WHSettings.RELATIVE, align_amount)
    return Align(align_type)