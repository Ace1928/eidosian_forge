from __future__ import annotations
import numba as nb
import numpy as np
import os
@arr_operator
def source_arr(src, dst):
    if src:
        return src
    else:
        return dst