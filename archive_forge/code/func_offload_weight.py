import json
import os
from collections.abc import Mapping
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from safetensors import safe_open
def offload_weight(weight, weight_name, offload_folder, index=None):
    dtype = None
    if str(weight.dtype) == 'torch.bfloat16':
        weight = weight.view(torch.int16)
        dtype = 'bfloat16'
    array = weight.cpu().numpy()
    tensor_file = os.path.join(offload_folder, f'{weight_name}.dat')
    if index is not None:
        if dtype is None:
            dtype = str(array.dtype)
        index[weight_name] = {'dtype': dtype, 'shape': list(array.shape)}
    if array.ndim == 0:
        array = array[None]
    file_array = np.memmap(tensor_file, dtype=array.dtype, mode='w+', shape=array.shape)
    file_array[:] = array[:]
    file_array.flush()
    return index