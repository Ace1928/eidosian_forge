from typing import List
import torch
def all_types_and(*dtypes):
    return _all_types + _validate_dtypes(*dtypes)