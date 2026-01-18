from __future__ import annotations
import dataclasses
import enum
import typing
def normalize_width(width: Literal['clip', 'pack', WHSettings.CLIP, WHSettings.PACK] | int | tuple[Literal['relative', WHSettings.RELATIVE], int] | tuple[Literal['weight', WHSettings.WEIGHT], int | float], err: type[BaseException]) -> tuple[Literal[WHSettings.CLIP, WHSettings.PACK], None] | tuple[Literal[WHSettings.GIVEN, WHSettings.RELATIVE], int] | tuple[Literal[WHSettings.WEIGHT], int | float]:
    """
    Split width into (width_type, width_amount).  Raise exception err
    if width doesn't match a valid alignment.
    """
    if width in {WHSettings.CLIP, WHSettings.PACK}:
        return (WHSettings(width), None)
    if isinstance(width, int):
        return (WHSettings.GIVEN, width)
    if isinstance(width, tuple) and len(width) == 2 and (width[0] in {WHSettings.RELATIVE, WHSettings.WEIGHT}):
        width_type, width_amount = width
        return (WHSettings(width_type), width_amount)
    raise err(f"width value {width!r} is not one offixed number of columns, 'pack', ('relative', percentage of total width), 'clip'")