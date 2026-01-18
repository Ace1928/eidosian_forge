from __future__ import annotations
from typing import TYPE_CHECKING, Any, cast
import numpy as np
from contourpy import FillType, LineType
from contourpy.enum_util import as_fill_type, as_line_type
from contourpy.types import MOVETO, code_dtype, offset_dtype, point_dtype
def check_filled(filled: cpy.FillReturn, fill_type: FillType | str) -> None:
    fill_type = as_fill_type(fill_type)
    if fill_type == FillType.OuterCode:
        if TYPE_CHECKING:
            filled = cast(cpy.FillReturn_OuterCode, filled)
        _check_tuple_of_lists_with_same_length(filled, 2)
        for i, (points, codes) in enumerate(zip(*filled)):
            check_point_array(points)
            check_code_array(codes)
            if len(points) != len(codes):
                raise ValueError(f'Points and codes have different lengths in polygon {i}')
    elif fill_type == FillType.OuterOffset:
        if TYPE_CHECKING:
            filled = cast(cpy.FillReturn_OuterOffset, filled)
        _check_tuple_of_lists_with_same_length(filled, 2)
        for i, (points, offsets) in enumerate(zip(*filled)):
            check_point_array(points)
            check_offset_array(offsets)
            if offsets[-1] != len(points):
                raise ValueError(f'Inconsistent points and offsets in polygon {i}')
    elif fill_type == FillType.ChunkCombinedCode:
        if TYPE_CHECKING:
            filled = cast(cpy.FillReturn_ChunkCombinedCode, filled)
        _check_tuple_of_lists_with_same_length(filled, 2, allow_empty_lists=False)
        for chunk, (points_or_none, codes_or_none) in enumerate(zip(*filled)):
            if points_or_none is not None and codes_or_none is not None:
                check_point_array(points_or_none)
                check_code_array(codes_or_none)
                if len(points_or_none) != len(codes_or_none):
                    raise ValueError(f'Points and codes have different lengths in chunk {chunk}')
            elif not (points_or_none is None and codes_or_none is None):
                raise ValueError(f'Inconsistent Nones in chunk {chunk}')
    elif fill_type == FillType.ChunkCombinedOffset:
        if TYPE_CHECKING:
            filled = cast(cpy.FillReturn_ChunkCombinedOffset, filled)
        _check_tuple_of_lists_with_same_length(filled, 2, allow_empty_lists=False)
        for chunk, (points_or_none, offsets_or_none) in enumerate(zip(*filled)):
            if points_or_none is not None and offsets_or_none is not None:
                check_point_array(points_or_none)
                check_offset_array(offsets_or_none)
                if offsets_or_none[-1] != len(points_or_none):
                    raise ValueError(f'Inconsistent points and offsets in chunk {chunk}')
            elif not (points_or_none is None and offsets_or_none is None):
                raise ValueError(f'Inconsistent Nones in chunk {chunk}')
    elif fill_type == FillType.ChunkCombinedCodeOffset:
        if TYPE_CHECKING:
            filled = cast(cpy.FillReturn_ChunkCombinedCodeOffset, filled)
        _check_tuple_of_lists_with_same_length(filled, 3, allow_empty_lists=False)
        for i, (points_or_none, codes_or_none, outer_offsets_or_none) in enumerate(zip(*filled)):
            if points_or_none is not None and codes_or_none is not None and (outer_offsets_or_none is not None):
                check_point_array(points_or_none)
                check_code_array(codes_or_none)
                check_offset_array(outer_offsets_or_none)
                if len(codes_or_none) != len(points_or_none):
                    raise ValueError(f'Points and codes have different lengths in chunk {i}')
                if outer_offsets_or_none[-1] != len(codes_or_none):
                    raise ValueError(f'Inconsistent codes and outer_offsets in chunk {i}')
            elif not (points_or_none is None and codes_or_none is None and (outer_offsets_or_none is None)):
                raise ValueError(f'Inconsistent Nones in chunk {i}')
    elif fill_type == FillType.ChunkCombinedOffsetOffset:
        if TYPE_CHECKING:
            filled = cast(cpy.FillReturn_ChunkCombinedOffsetOffset, filled)
        _check_tuple_of_lists_with_same_length(filled, 3, allow_empty_lists=False)
        for i, (points_or_none, offsets_or_none, outer_offsets_or_none) in enumerate(zip(*filled)):
            if points_or_none is not None and offsets_or_none is not None and (outer_offsets_or_none is not None):
                check_point_array(points_or_none)
                check_offset_array(offsets_or_none)
                check_offset_array(outer_offsets_or_none)
                if offsets_or_none[-1] != len(points_or_none):
                    raise ValueError(f'Inconsistent points and offsets in chunk {i}')
                if outer_offsets_or_none[-1] != len(offsets_or_none) - 1:
                    raise ValueError(f'Inconsistent offsets and outer_offsets in chunk {i}')
            elif not (points_or_none is None and offsets_or_none is None and (outer_offsets_or_none is None)):
                raise ValueError(f'Inconsistent Nones in chunk {i}')
    else:
        raise ValueError(f'Invalid FillType {fill_type}')