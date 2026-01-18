import itertools
import math
from typing import Sequence, Tuple, Union
import numpy as np
from onnx.reference.op_run import OpRun
def get_output_shape_auto_pad(auto_pad: str, input_spatial_shape: Sequence[int], kernel_spatial_shape: Sequence[int], strides_spatial: Sequence[int]) -> Sequence[int]:
    """https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling2D
    output_shape = math.floor((input_shape - 1) / strides) + 1  (SAME)
    output_shape = math.floor((input_shape - pool_size) / strides) + 1 (VALID)
    IMPORTANT: this function assumes ceil_mode is False. In tenforflow, ceil_mode is always False.
    However, ONNX spec allow ceil_mode to be True because ORT does handle the case.
    """
    strides_spatial = strides_spatial or [1] * len(input_spatial_shape)
    out_shape = [0] * len(input_spatial_shape)
    for i in range(len(input_spatial_shape)):
        if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
            out_shape[i] = math.floor((input_spatial_shape[i] - 1) / strides_spatial[i]) + 1
        elif auto_pad == 'VALID':
            out_shape[i] = math.floor((input_spatial_shape[i] - kernel_spatial_shape[i]) / strides_spatial[i]) + 1
        else:
            raise ValueError('auto_pad can only be NOTSET, SAME_UPPER, SAME_LOWER, or VALID')
    return out_shape