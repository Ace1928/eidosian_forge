from __future__ import annotations
from typing import TYPE_CHECKING, cast
import matplotlib.path as mpath
import numpy as np
from contourpy import FillType, LineType
from contourpy.array import codes_from_offsets
def lines_to_mpl_paths(lines: LineReturn, line_type: LineType) -> list[mpath.Path]:
    if line_type == LineType.Separate:
        if TYPE_CHECKING:
            lines = cast(LineReturn_Separate, lines)
        paths = []
        for line in lines:
            closed = line[0, 0] == line[-1, 0] and line[0, 1] == line[-1, 1]
            paths.append(mpath.Path(line, closed=closed))
    elif line_type in (LineType.SeparateCode, LineType.ChunkCombinedCode):
        paths = [mpath.Path(points, codes) for points, codes in zip(*lines) if points is not None]
    elif line_type == LineType.ChunkCombinedOffset:
        paths = []
        for points, offsets in zip(*lines):
            if points is None:
                continue
            for i in range(len(offsets) - 1):
                line = points[offsets[i]:offsets[i + 1]]
                closed = line[0, 0] == line[-1, 0] and line[0, 1] == line[-1, 1]
                paths.append(mpath.Path(line, closed=closed))
    elif line_type == LineType.ChunkCombinedNan:
        paths = []
        for points in lines[0]:
            if points is None:
                continue
            nan_offsets = np.nonzero(np.isnan(points[:, 0]))[0]
            nan_offsets = np.concatenate([[-1], nan_offsets, [len(points)]])
            for s, e in zip(nan_offsets[:-1], nan_offsets[1:]):
                line = points[s + 1:e]
                closed = line[0, 0] == line[-1, 0] and line[0, 1] == line[-1, 1]
                paths.append(mpath.Path(line, closed=closed))
    else:
        raise RuntimeError(f'Conversion of LineType {line_type} to MPL Paths is not implemented')
    return paths