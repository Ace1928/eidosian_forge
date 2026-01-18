from __future__ import annotations
from typing import TYPE_CHECKING, cast
from contourpy._contourpy import FillType, LineType
from contourpy.array import (
from contourpy.enum_util import as_fill_type, as_line_type
from contourpy.typecheck import check_filled, check_lines
def dechunk_filled(filled: cpy.FillReturn, fill_type: FillType | str) -> cpy.FillReturn:
    """Return the specified filled contours with all chunked data moved into the first chunk.

    Filled contours that are not chunked (``FillType.OuterCode`` and ``FillType.OuterOffset``) and
    those that are but only contain a single chunk are returned unmodified. Individual polygons are
    unchanged, they are not geometrically combined.

    Args:
        filled (sequence of arrays): Filled contour data as returned by
            :func:`~contourpy.ContourGenerator.filled`.
        fill_type (FillType or str): Type of ``filled`` as enum or string equivalent.

    Return:
        Filled contours in a single chunk.

    .. versionadded:: 1.2.0
    """
    fill_type = as_fill_type(fill_type)
    if fill_type in (FillType.OuterCode, FillType.OuterOffset):
        return filled
    check_filled(filled, fill_type)
    if len(filled[0]) < 2:
        return filled
    if TYPE_CHECKING:
        filled = cast(cpy.FillReturn_Chunk, filled)
    points = concat_points_or_none(filled[0])
    if fill_type == FillType.ChunkCombinedCode:
        if TYPE_CHECKING:
            filled = cast(cpy.FillReturn_ChunkCombinedCode, filled)
        if points is None:
            ret1: cpy.FillReturn_ChunkCombinedCode = ([None], [None])
        else:
            ret1 = ([points], [concat_codes_or_none(filled[1])])
        return ret1
    elif fill_type == FillType.ChunkCombinedOffset:
        if TYPE_CHECKING:
            filled = cast(cpy.FillReturn_ChunkCombinedOffset, filled)
        if points is None:
            ret2: cpy.FillReturn_ChunkCombinedOffset = ([None], [None])
        else:
            ret2 = ([points], [concat_offsets_or_none(filled[1])])
        return ret2
    elif fill_type == FillType.ChunkCombinedCodeOffset:
        if TYPE_CHECKING:
            filled = cast(cpy.FillReturn_ChunkCombinedCodeOffset, filled)
        if points is None:
            ret3: cpy.FillReturn_ChunkCombinedCodeOffset = ([None], [None], [None])
        else:
            outer_offsets = concat_offsets_or_none(filled[2])
            ret3 = ([points], [concat_codes_or_none(filled[1])], [outer_offsets])
        return ret3
    elif fill_type == FillType.ChunkCombinedOffsetOffset:
        if TYPE_CHECKING:
            filled = cast(cpy.FillReturn_ChunkCombinedOffsetOffset, filled)
        if points is None:
            ret4: cpy.FillReturn_ChunkCombinedOffsetOffset = ([None], [None], [None])
        else:
            outer_offsets = concat_offsets_or_none(filled[2])
            ret4 = ([points], [concat_offsets_or_none(filled[1])], [outer_offsets])
        return ret4
    else:
        raise ValueError(f'Invalid FillType {fill_type}')