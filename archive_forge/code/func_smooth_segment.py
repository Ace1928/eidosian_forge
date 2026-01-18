from __future__ import annotations
from math import ceil
from dask import compute, delayed
from pandas import DataFrame
import numpy as np
import pandas as pd
import param
from .utils import ngjit
@ngjit
def smooth_segment(segments, tension, idx, idy):
    seg_length = len(segments) - 2
    for i in range(1, seg_length):
        previous, current, next_point = (segments[i - 1], segments[i], segments[i + 1])
        current[idx] = (1 - tension) * current[idx] + tension * (previous[idx] + next_point[idx]) / 2
        current[idy] = (1 - tension) * current[idy] + tension * (previous[idy] + next_point[idy]) / 2