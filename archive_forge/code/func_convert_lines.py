from __future__ import annotations
from typing import TYPE_CHECKING, cast
import numpy as np
from contourpy._contourpy import FillType, LineType
import contourpy.array as arr
from contourpy.enum_util import as_fill_type, as_line_type
from contourpy.typecheck import check_filled, check_lines
from contourpy.types import MOVETO, offset_dtype
def convert_lines(lines: cpy.LineReturn, line_type_from: LineType | str, line_type_to: LineType | str) -> cpy.LineReturn:
    """Return the specified contour lines converted to a different :class:`~contourpy.LineType`.

    Args:
        lines (sequence of arrays): Contour lines to convert.
        line_type_from (LineType or str): :class:`~contourpy.LineType` to convert from as enum or
            string equivalent.
        line_type_to (LineType or str): :class:`~contourpy.LineType` to convert to as enum or string
            equivalent.

    Return:
        Converted contour lines.

    When converting non-chunked line types (``LineType.Separate`` or ``LineType.SeparateCode``) to
    chunked ones (``LineType.ChunkCombinedCode``, ``LineType.ChunkCombinedOffset`` or
    ``LineType.ChunkCombinedNan``), all lines are placed in the first chunk. When converting in the
    other direction, all chunk information is discarded.

    .. versionadded:: 1.2.0
    """
    line_type_from = as_line_type(line_type_from)
    line_type_to = as_line_type(line_type_to)
    check_lines(lines, line_type_from)
    if line_type_from == LineType.Separate:
        if TYPE_CHECKING:
            lines = cast(cpy.LineReturn_Separate, lines)
        return _convert_lines_from_Separate(lines, line_type_to)
    elif line_type_from == LineType.SeparateCode:
        if TYPE_CHECKING:
            lines = cast(cpy.LineReturn_SeparateCode, lines)
        return _convert_lines_from_SeparateCode(lines, line_type_to)
    elif line_type_from == LineType.ChunkCombinedCode:
        if TYPE_CHECKING:
            lines = cast(cpy.LineReturn_ChunkCombinedCode, lines)
        return _convert_lines_from_ChunkCombinedCode(lines, line_type_to)
    elif line_type_from == LineType.ChunkCombinedOffset:
        if TYPE_CHECKING:
            lines = cast(cpy.LineReturn_ChunkCombinedOffset, lines)
        return _convert_lines_from_ChunkCombinedOffset(lines, line_type_to)
    elif line_type_from == LineType.ChunkCombinedNan:
        if TYPE_CHECKING:
            lines = cast(cpy.LineReturn_ChunkCombinedNan, lines)
        return _convert_lines_from_ChunkCombinedNan(lines, line_type_to)
    else:
        raise ValueError(f'Invalid LineType {line_type_from}')