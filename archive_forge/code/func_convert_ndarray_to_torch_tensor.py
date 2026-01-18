import os
import warnings
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
import torch
import ray
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed
def convert_ndarray_to_torch_tensor(ndarray: np.ndarray, dtype: Optional[torch.dtype]=None, device: Optional[str]=None) -> torch.Tensor:
    """Convert a NumPy ndarray to a Torch Tensor.

    Args:
        ndarray: A NumPy ndarray that we wish to convert to a Torch Tensor.
        dtype: A Torch dtype for the created tensor; if None, the dtype will be
            inferred from the NumPy ndarray data.
        device: The device on which the tensor(s) should be placed; if None, the Torch
            tensor(s) will be constructed on the CPU.

    Returns: A Torch Tensor.
    """
    ndarray = _unwrap_ndarray_object_type_if_needed(ndarray)
    if ndarray.dtype.type is np.object_:
        raise RuntimeError('Numpy array of object dtype cannot be converted to a Torch Tensor. This may because the numpy array is a ragged tensor--it contains items of different sizes. If using `iter_torch_batches()` API, you can pass in a `collate_fn` argument to specify custom logic to convert the Numpy array batch to a Torch tensor batch.')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return torch.as_tensor(ndarray, dtype=dtype, device=device)