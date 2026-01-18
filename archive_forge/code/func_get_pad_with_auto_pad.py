import itertools
import math
from typing import Sequence, Tuple, Union
import numpy as np
from onnx.reference.op_run import OpRun
def get_pad_with_auto_pad(auto_pad: str, pad_shape: Sequence[int]) -> Sequence[int]:
    spatial_dims = len(pad_shape)
    if auto_pad == 'SAME_UPPER':
        pads = [pad_shape[i] // 2 for i in range(spatial_dims)] + [pad_shape[i] - pad_shape[i] // 2 for i in range(spatial_dims)]
    elif auto_pad == 'SAME_LOWER':
        pads = [pad_shape[i] - pad_shape[i] // 2 for i in range(spatial_dims)] + [pad_shape[i] // 2 for i in range(spatial_dims)]
    else:
        pads = [0] * spatial_dims * 2
    return pads