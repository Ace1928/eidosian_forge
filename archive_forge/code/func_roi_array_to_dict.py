from datetime import datetime
import logging
import os
from typing import (
import warnings
import numpy as np
from ..core.request import Request, IOMode, InitializationError
from ..core.v3_plugin_api import PluginV3, ImageProperties
def roi_array_to_dict(a: np.ndarray) -> List[Dict[str, List[int]]]:
    """Convert the `ROIs` structured arrays to :py:class:`dict`

    Parameters
    ----------
    a
        Structured array containing ROI data

    Returns
    -------
    One dict per ROI. Keys are "top_left", "bottom_right", and "bin",
    values are tuples whose first element is the x axis value and the
    second element is the y axis value.
    """
    dict_list = []
    a = a[['startx', 'starty', 'endx', 'endy', 'groupx', 'groupy']]
    for sx, sy, ex, ey, gx, gy in a:
        roi_dict = {'top_left': [int(sx), int(sy)], 'bottom_right': [int(ex), int(ey)], 'bin': [int(gx), int(gy)]}
        dict_list.append(roi_dict)
    return dict_list