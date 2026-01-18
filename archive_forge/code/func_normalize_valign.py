from __future__ import annotations
import dataclasses
import enum
import typing
def normalize_valign(valign: Literal['top', 'middle', 'bottom'] | VAlign | tuple[Literal['relative', WHSettings.RELATIVE], int], err: type[BaseException]) -> tuple[VAlign, None] | tuple[Literal[WHSettings.RELATIVE], int]:
    """
    Split align into (valign_type, valign_amount).  Raise exception err
    if align doesn't match a valid alignment.
    """
    if valign in {VAlign.TOP, VAlign.MIDDLE, VAlign.BOTTOM}:
        return (VAlign(valign), None)
    if isinstance(valign, tuple) and len(valign) == 2 and (valign[0] == WHSettings.RELATIVE):
        _valign_type, valign_amount = valign
        return (WHSettings.RELATIVE, valign_amount)
    raise err(f"valign value {valign!r} is not one of 'top', 'middle', 'bottom', ('relative', percentage 0=left 100=right)")