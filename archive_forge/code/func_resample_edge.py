from __future__ import annotations
from math import ceil
from dask import compute, delayed
from pandas import DataFrame
import numpy as np
import pandas as pd
import param
from .utils import ngjit
def resample_edge(segments, min_segment_length, max_segment_length, ndims):
    change, total_resamples = calculate_length(segments, min_segment_length, max_segment_length)
    if not change:
        return segments
    resampled = np.empty((total_resamples, ndims))
    resample_segment(segments, resampled, min_segment_length, max_segment_length, ndims)
    return resampled