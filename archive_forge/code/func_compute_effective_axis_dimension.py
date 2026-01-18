from ctypes import c_float, sizeof
from enum import Enum
from typing import TYPE_CHECKING, Optional, Union
def compute_effective_axis_dimension(dimension: int, fixed_dimension: int, num_token_to_add: int=0) -> int:
    """

    Args:
        dimension:
        fixed_dimension:
        num_token_to_add:

    Returns:

    """
    if dimension <= 0:
        dimension = fixed_dimension
    dimension -= num_token_to_add
    return dimension