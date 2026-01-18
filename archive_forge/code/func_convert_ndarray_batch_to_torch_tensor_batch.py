import os
import warnings
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
import torch
import ray
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed
def convert_ndarray_batch_to_torch_tensor_batch(ndarrays: Union[np.ndarray, Dict[str, np.ndarray]], dtypes: Optional[Union[torch.dtype, Dict[str, torch.dtype]]]=None, device: Optional[str]=None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Convert a NumPy ndarray batch to a Torch Tensor batch.

    Args:
        ndarray: A (dict of) NumPy ndarray(s) that we wish to convert to a Torch Tensor.
        dtype: A (dict of) Torch dtype(s) for the created tensor; if None, the dtype
            will be inferred from the NumPy ndarray data.
        device: The device on which the tensor(s) should be placed; if None, the Torch
            tensor(s) will be constructed on the CPU.

    Returns: A (dict of) Torch Tensor(s).
    """
    if isinstance(ndarrays, np.ndarray):
        if isinstance(dtypes, dict):
            if len(dtypes) != 1:
                raise ValueError(f'When constructing a single-tensor batch, only a single dtype should be given, instead got: {dtypes}')
            dtypes = next(iter(dtypes.values()))
        batch = convert_ndarray_to_torch_tensor(ndarrays, dtype=dtypes, device=device)
    else:
        batch = {col_name: convert_ndarray_to_torch_tensor(col_ndarray, dtype=dtypes[col_name] if isinstance(dtypes, dict) else dtypes, device=device) for col_name, col_ndarray in ndarrays.items()}
    return batch