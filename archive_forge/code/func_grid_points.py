from __future__ import annotations
import math
import re
import numpy as np
def grid_points(chunks, sizes):
    cumchunks = [np.cumsum((0,) + c) for c in chunks]
    points = [x * size / x[-1] for x, size in zip(cumchunks, sizes)]
    return points