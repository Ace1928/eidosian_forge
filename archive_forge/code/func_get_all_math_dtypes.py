from typing import List
import torch
def get_all_math_dtypes(device) -> List[torch.dtype]:
    return get_all_int_dtypes() + get_all_fp_dtypes(include_half=device.startswith('cuda'), include_bfloat16=False) + get_all_complex_dtypes()