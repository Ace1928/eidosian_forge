from ctypes import c_float, sizeof
from enum import Enum
from typing import TYPE_CHECKING, Optional, Union
def compute_serialized_parameters_size(num_parameters: int, dtype: ParameterFormat) -> int:
    """
    Compute the size taken by all the parameters in the given the storage format when serializing the model

    Args:
        num_parameters: Number of parameters to be saved
        dtype: The data format each parameter will be saved

    Returns:
        Size (in byte) taken to save all the parameters
    """
    return num_parameters * dtype.size