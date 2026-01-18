import copy
import io
from typing import List, Union
import torch
def safetensors_load_file_wrapper(filename, device='cpu'):
    result = {}
    with safetensors.torch.safe_open(filename, framework='pt', device=device) as f:
        for k in f.keys():
            fake_mode = torch._guards.detect_fake_mode()
            if not fake_mode:
                result[k] = f.get_tensor(k)
            else:
                empty_tensor = f.get_slice(k)
                result[k] = torch.empty(tuple(empty_tensor.get_shape()), dtype=safetensors.torch._getdtype(empty_tensor.get_dtype()))
    return result