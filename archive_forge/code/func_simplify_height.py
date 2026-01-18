from __future__ import annotations
import dataclasses
import enum
import typing
def simplify_height(height_type: Literal['flow', 'pack', 'relative', 'given', 'weight', WHSettings.FLOW, WHSettings.PACK, WHSettings.RELATIVE, WHSettings.GIVEN, WHSettings.WEIGHT], height_amount: int | float | None) -> int | Literal[WHSettings.FLOW, WHSettings.PACK] | tuple[Literal[WHSettings.RELATIVE], int] | tuple[Literal[WHSettings.WEIGHT], int | float]:
    """
    Recombine (height_type, height_amount) into a height value.
    Inverse of normalize_height.
    """
    if height_type in {WHSettings.FLOW, WHSettings.PACK}:
        return WHSettings(height_type)
    if not isinstance(height_amount, int):
        raise TypeError(height_amount)
    if height_type == WHSettings.GIVEN:
        return height_amount
    return (WHSettings(height_type), height_amount)