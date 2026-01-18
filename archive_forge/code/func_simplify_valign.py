from __future__ import annotations
import dataclasses
import enum
import typing
def simplify_valign(valign_type: Literal['top', 'middle', 'bottom', 'relative', WHSettings.RELATIVE] | VAlign, valign_amount: int | None) -> VAlign | tuple[Literal[WHSettings.RELATIVE], int]:
    """
    Recombine (valign_type, valign_amount) into an valign value.
    Inverse of normalize_valign.
    """
    if valign_type == WHSettings.RELATIVE:
        if not isinstance(valign_amount, int):
            raise TypeError(valign_amount)
        return (WHSettings.RELATIVE, valign_amount)
    return VAlign(valign_type)