from typing import List
import torch
def all_types_and_complex_and(*dtypes):
    return _all_types_and_complex + _validate_dtypes(*dtypes)