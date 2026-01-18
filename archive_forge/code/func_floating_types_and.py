from typing import List
import torch
def floating_types_and(*dtypes):
    return _floating_types + _validate_dtypes(*dtypes)