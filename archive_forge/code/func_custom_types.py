from typing import List
import torch
def custom_types(*dtypes):
    """Create a list of arbitrary dtypes"""
    return _empty_types + _validate_dtypes(*dtypes)