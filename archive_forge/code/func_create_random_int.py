import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from onnx import MapProto, OptionalProto, SequenceProto, TensorProto, helper, subbyte
from onnx.external_data_helper import load_external_data_for_tensor, uses_external_data
def create_random_int(input_shape: Tuple[int], dtype: np.dtype, seed: int=1) -> np.ndarray:
    """Create random integer array for backend/test/case/node.

    Args:
        input_shape: The shape for the returned integer array.
        dtype: The NumPy data type for the returned integer array.
        seed: The seed for np.random.

    Returns:
        np.ndarray: Random integer array.
    """
    np.random.seed(seed)
    if dtype in (np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64):
        end = min(np.iinfo(dtype).max, np.iinfo(np.int32).max)
        start = max(np.iinfo(dtype).min, np.iinfo(np.int32).min)
        return np.random.randint(start, end, size=input_shape).astype(dtype)
    else:
        raise TypeError(f'{dtype} is not supported by create_random_int.')