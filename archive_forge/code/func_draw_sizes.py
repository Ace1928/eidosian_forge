from __future__ import annotations
import math
import re
import numpy as np
def draw_sizes(shape, size=200):
    """Get size in pixels for all dimensions"""
    mx = max(shape)
    ratios = [mx / max(0.1, d) for d in shape]
    ratios = [ratio_response(r) for r in ratios]
    return tuple((size / r for r in ratios))