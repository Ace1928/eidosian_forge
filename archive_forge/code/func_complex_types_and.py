from typing import List
import torch
def complex_types_and(*dtypes):
    return _complex_types + _validate_dtypes(*dtypes)