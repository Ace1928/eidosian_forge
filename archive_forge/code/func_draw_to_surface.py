from __future__ import annotations
from math import ceil
from dask import compute, delayed
from pandas import DataFrame
import numpy as np
import pandas as pd
import param
from .utils import ngjit
@delayed
def draw_to_surface(edge_segments, bandwidth, accuracy, accumulator):
    img = np.zeros((accuracy + 1, accuracy + 1))
    for segments in edge_segments:
        for point in segments:
            accumulator(img, point, accuracy)
    return gaussian(img, sigma=bandwidth / 2)