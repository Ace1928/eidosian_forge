from __future__ import annotations
import logging # isort:skip
from typing import Any
import numpy as np
from .dependencies import import_required
def axial_to_cartesian(q: Any, r: Any, size: float, orientation: str, aspect_scale: float=1) -> tuple[Any, Any]:
    """ Map axial *(q,r)* coordinates to cartesian *(x,y)* coordinates of
    tiles centers.

    This function can be useful for positioning other Bokeh glyphs with
    cartesian coordinates in relation to a hex tiling.

    This function was adapted from:

    https://www.redblobgames.com/grids/hexagons/#hex-to-pixel

    Args:
        q (array[float]) :
            A NumPy array of q-coordinates for binning

        r (array[float]) :
            A NumPy array of r-coordinates for binning

        size (float) :
            The size of the hexagonal tiling.

            The size is defined as the distance from the center of a hexagon
            to the top corner for "pointytop" orientation, or from the center
            to a side corner for "flattop" orientation.

        orientation (str) :
            Whether the hex tile orientation should be "pointytop" or
            "flattop".

        aspect_scale (float, optional) :
            Scale the hexagons in the "cross" dimension.

            For "pointytop" orientations, hexagons are scaled in the horizontal
            direction. For "flattop", they are scaled in vertical direction.

            When working with a plot with ``aspect_scale != 1``, it may be
            useful to set this value to match the plot.

    Returns:
        (array[int], array[int])

    """
    if orientation == 'pointytop':
        x = size * np.sqrt(3) * (q + r / 2.0) / aspect_scale
        y = -size * 3 / 2.0 * r
    else:
        x = size * 3 / 2.0 * q
        y = -size * np.sqrt(3) * (r + q / 2.0) * aspect_scale
    return (x, y)