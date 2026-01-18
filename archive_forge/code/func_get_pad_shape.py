import itertools
import math
from typing import Sequence, Tuple, Union
import numpy as np
from onnx.reference.op_run import OpRun
def get_pad_shape(auto_pad: str, input_spatial_shape: Sequence[int], kernel_spatial_shape: Sequence[int], strides_spatial: Sequence[int], output_spatial_shape: Sequence[int]) -> Sequence[int]:
    spatial_dims = len(input_spatial_shape)
    pad_shape = [0] * spatial_dims
    strides_spatial = strides_spatial or [1] * spatial_dims
    if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
        for i in range(spatial_dims):
            pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
    elif auto_pad == 'VALID':
        pass
    return pad_shape