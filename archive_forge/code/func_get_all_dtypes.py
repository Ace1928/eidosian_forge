from typing import List
import torch
def get_all_dtypes(include_half=True, include_bfloat16=True, include_bool=True, include_complex=True, include_complex32=False, include_qint=False) -> List[torch.dtype]:
    dtypes = get_all_int_dtypes() + get_all_fp_dtypes(include_half=include_half, include_bfloat16=include_bfloat16)
    if include_bool:
        dtypes.append(torch.bool)
    if include_complex:
        dtypes += get_all_complex_dtypes(include_complex32)
    if include_qint:
        dtypes += get_all_qint_dtypes()
    return dtypes