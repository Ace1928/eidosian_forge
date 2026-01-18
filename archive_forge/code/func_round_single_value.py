import numbers
from typing import List
import numpy as np
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_resize import _get_all_coords
def round_single_value(v):
    if v >= 0.0:
        return np.floor(v + 0.5)
    else:
        return np.ceil(v - 0.5)