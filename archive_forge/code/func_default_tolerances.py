import abc
import cmath
import collections.abc
import contextlib
import warnings
from typing import (
import torch
def default_tolerances(*inputs: Union[torch.Tensor, torch.dtype], dtype_precisions: Optional[Dict[torch.dtype, Tuple[float, float]]]=None) -> Tuple[float, float]:
    """Returns the default absolute and relative testing tolerances for a set of inputs based on the dtype.

    See :func:`assert_close` for a table of the default tolerance for each dtype.

    Returns:
        (Tuple[float, float]): Loosest tolerances of all input dtypes.
    """
    dtypes = []
    for input in inputs:
        if isinstance(input, torch.Tensor):
            dtypes.append(input.dtype)
        elif isinstance(input, torch.dtype):
            dtypes.append(input)
        else:
            raise TypeError(f'Expected a torch.Tensor or a torch.dtype, but got {type(input)} instead.')
    dtype_precisions = dtype_precisions or _DTYPE_PRECISIONS
    rtols, atols = zip(*[dtype_precisions.get(dtype, (0.0, 0.0)) for dtype in dtypes])
    return (max(rtols), max(atols))