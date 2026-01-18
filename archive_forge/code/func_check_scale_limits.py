from __future__ import annotations
import typing
from copy import copy
import pandas as pd
from matplotlib.animation import ArtistAnimation
from .exceptions import PlotnineError
def check_scale_limits(scales: list[scale], frame_no: int):
    """
            Check limits of the scales of a plot in the animation

            Raises a PlotnineError if any of the scales has limits
            that do not match those of the first plot/frame.

            Should be called after `set_scale_limits`.

            Parameters
            ----------
            scales : list[scales]
                List of scales the have been used in building a
                ggplot object.

            frame_no : int
                Frame number
            """
    if len(scale_limits) != len(scales):
        raise PlotnineError('All plots must have the same number of scales as the first plot of the animation.')
    for sc in scales:
        ae = sc.aesthetics[0]
        if ae not in scale_limits:
            raise PlotnineError(f'The plot for frame {frame_no} does not have a scale for the {ae} aesthetic.')
        if sc.limits != scale_limits[ae]:
            raise PlotnineError(f'The {ae} scale of plot for frame {frame_no} has different limits from those of the first frame.')