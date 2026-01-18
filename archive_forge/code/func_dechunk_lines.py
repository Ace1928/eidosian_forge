from __future__ import annotations
from typing import TYPE_CHECKING, cast
from contourpy._contourpy import FillType, LineType
from contourpy.array import (
from contourpy.enum_util import as_fill_type, as_line_type
from contourpy.typecheck import check_filled, check_lines
def dechunk_lines(lines: cpy.LineReturn, line_type: LineType | str) -> cpy.LineReturn:
    """Return the specified contour lines with all chunked data moved into the first chunk.

    Contour lines that are not chunked (``LineType.Separate`` and ``LineType.SeparateCode``) and
    those that are but only contain a single chunk are returned unmodified. Individual lines are
    unchanged, they are not geometrically combined.

    Args:
        lines (sequence of arrays): Contour line data as returned by
            :func:`~contourpy.ContourGenerator.lines`.
        line_type (LineType or str): Type of ``lines`` as enum or string equivalent.

    Return:
        Contour lines in a single chunk.

    .. versionadded:: 1.2.0
    """
    line_type = as_line_type(line_type)
    if line_type in (LineType.Separate, LineType.SeparateCode):
        return lines
    check_lines(lines, line_type)
    if len(lines[0]) < 2:
        return lines
    if TYPE_CHECKING:
        lines = cast(cpy.LineReturn_Chunk, lines)
    if line_type == LineType.ChunkCombinedCode:
        if TYPE_CHECKING:
            lines = cast(cpy.LineReturn_ChunkCombinedCode, lines)
        points = concat_points_or_none(lines[0])
        if points is None:
            ret1: cpy.LineReturn_ChunkCombinedCode = ([None], [None])
        else:
            ret1 = ([points], [concat_codes_or_none(lines[1])])
        return ret1
    elif line_type == LineType.ChunkCombinedOffset:
        if TYPE_CHECKING:
            lines = cast(cpy.LineReturn_ChunkCombinedOffset, lines)
        points = concat_points_or_none(lines[0])
        if points is None:
            ret2: cpy.LineReturn_ChunkCombinedOffset = ([None], [None])
        else:
            ret2 = ([points], [concat_offsets_or_none(lines[1])])
        return ret2
    elif line_type == LineType.ChunkCombinedNan:
        if TYPE_CHECKING:
            lines = cast(cpy.LineReturn_ChunkCombinedNan, lines)
        points = concat_points_or_none_with_nan(lines[0])
        ret3: cpy.LineReturn_ChunkCombinedNan = ([points],)
        return ret3
    else:
        raise ValueError(f'Invalid LineType {line_type}')