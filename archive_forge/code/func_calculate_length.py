from __future__ import annotations
from math import ceil
from dask import compute, delayed
from pandas import DataFrame
import numpy as np
import pandas as pd
import param
from .utils import ngjit
@ngjit
def calculate_length(segments, min_segment_length, max_segment_length):
    current_point = segments[0]
    index = 1
    total = 0
    any_change = False
    while index < len(segments):
        next_point = segments[index]
        distance = distance_between(current_point, next_point)
        if distance < min_segment_length and 1 < index < len(segments) - 2:
            any_change = True
            current_point = (current_point + next_point) / 2
            total += 1
            index += 2
        elif distance > max_segment_length:
            any_change = True
            points = int(ceil(distance / ((max_segment_length + min_segment_length) / 2)))
            total += points
            current_point = next_point
            index += 1
        else:
            total += 1
            current_point = next_point
            index += 1
    total += 1
    return (any_change, total)