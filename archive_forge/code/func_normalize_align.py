from __future__ import annotations
import dataclasses
import enum
import typing
def normalize_align(align: Literal['left', 'center', 'right'] | Align | tuple[Literal['relative', WHSettings.RELATIVE], int], err: type[BaseException]) -> tuple[Align, None] | tuple[Literal[WHSettings.RELATIVE], int]:
    """
    Split align into (align_type, align_amount).  Raise exception err
    if align doesn't match a valid alignment.
    """
    if align in {Align.LEFT, Align.CENTER, Align.RIGHT}:
        return (Align(align), None)
    if isinstance(align, tuple) and len(align) == 2 and (align[0] == WHSettings.RELATIVE):
        _align_type, align_amount = align
        return (WHSettings.RELATIVE, align_amount)
    raise err(f"align value {align!r} is not one of 'left', 'center', 'right', ('relative', percentage 0=left 100=right)")