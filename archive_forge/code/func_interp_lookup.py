from __future__ import annotations
import typing
from dataclasses import dataclass
import numpy as np
from ..hsluv import hex_to_rgb, rgb_to_hex
from ._colormap import ColorMap, ColorMapKind
def interp_lookup(x: NDArrayFloat, values: NDArrayFloat, values_alt: Optional[NDArrayFloat]=None) -> NDArrayFloat:
    """
    Create an interpolation lookup array

    This helps make interpolating between two or more colors
    a discrete task.

    Parameters
    ----------
    x:
        Breaks In the range [0, 1]. Must include 0 and 1 and values
        should be sorted.
    values:
        In the range [0, 1]. Must be the same length as x.
    values_alt:
        In the range [0, 1]. Must be the same length as x.
        Makes it possible to have adjacent interpolation regions
        that with gaps in them numbers. e.g.

            values = [0, 0.1, 0.5, 1]
            values_alt = [0, 0.1, 0.6, 1]

        Creates the regions

            [(0, 0.1), (0.1, 0.5), (0.6, 1)]

        If values_alt is None the region would be

            [(0, 0.1), (0.1, 0.5), (0.5, 1)]
    """
    if values_alt is None:
        values_alt = values
    x256 = x * 255
    ind = np.searchsorted(x256, SPACE256)[1:-1]
    ind_prev = ind - 1
    distance = (INNER_SPACE256 - x256[ind_prev]) / (x256[ind] - x256[ind_prev])
    stop = values[ind]
    start = values_alt[ind_prev]
    lut = np.concatenate([[values[0]], start + distance * (stop - start), [values[-1]]])
    return np.clip(lut, 0, 1)