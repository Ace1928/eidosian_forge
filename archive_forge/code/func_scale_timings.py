from copy import deepcopy
import csv, math, os
from nibabel import load
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import ensure_list
from ..utils.misc import normalize_mc_params
from .. import config, logging
def scale_timings(timelist, input_units, output_units, time_repetition):
    """
    Scale timings given input and output units (scans/secs).

    Parameters
    ----------
    timelist: list of times to scale
    input_units: 'secs' or 'scans'
    output_units: Ibid.
    time_repetition: float in seconds

    """
    if input_units == output_units:
        _scalefactor = 1.0
    if input_units == 'scans' and output_units == 'secs':
        _scalefactor = time_repetition
    if input_units == 'secs' and output_units == 'scans':
        _scalefactor = 1.0 / time_repetition
    timelist = [np.max([0.0, _scalefactor * t]) for t in timelist]
    return timelist