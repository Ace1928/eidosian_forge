from __future__ import annotations
from math import ceil
from dask import compute, delayed
from pandas import DataFrame
import numpy as np
import pandas as pd
import param
from .utils import ngjit
@ngjit
def resample_segment(segments, new_segments, min_segment_length, max_segment_length, ndims):
    next_point = np.zeros(ndims, dtype=segments.dtype)
    current_point = segments[0]
    pos = 0
    index = 1
    while index < len(segments):
        next_point = segments[index]
        distance = distance_between(current_point, next_point)
        if distance < min_segment_length and 1 < index < len(segments) - 2:
            current_point = (current_point + next_point) / 2
            new_segments[pos] = current_point
            pos += 1
            index += 2
        elif distance > max_segment_length:
            points = int(ceil(distance / ((max_segment_length + min_segment_length) / 2)))
            for i in range(points):
                new_segments[pos] = current_point + i * ((next_point - current_point) / points)
                pos += 1
            current_point = next_point
            index += 1
        else:
            new_segments[pos] = current_point
            pos += 1
            current_point = next_point
            index += 1
    new_segments[pos] = next_point
    return new_segments