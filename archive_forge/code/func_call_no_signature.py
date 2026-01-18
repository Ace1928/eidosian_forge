import asyncio
import builtins
import functools
import inspect
from typing import Callable, Optional
import numpy as np
from numpy.lib.function_base import (
def call_no_signature(self, *args, **kwargs):
    """Call functions and coroutines when no signature is specified.

        When no signature is specified we assume that all of the function's
        inputs and outputs are scalars (core dimension of zero). We first
        broadcast the input arrays, then iteratively apply the function over the
        elements of the broadcasted arrays and finally reshape the results to
        match the input shape.

        Functions are executed in a for loop, coroutines are executed
        concurrently.

        """
    args = [np.array(arg) for arg in args]
    kwargs = {key: np.array(value) for key, value in kwargs.items()}
    broadcast_shape = np.broadcast(*args, *list(kwargs.values())).shape
    args = [np.broadcast_to(arg, broadcast_shape) for arg in args]
    kwargs = {key: np.broadcast_to(value, broadcast_shape) for key, value in kwargs.items()}
    if self.is_coroutine_fn:
        outputs = self.vectorize_call_coroutine(broadcast_shape, args, kwargs)
    else:
        outputs = self.vectorize_call(broadcast_shape, args, kwargs)
    outputs = [results if isinstance(results, tuple) else (results,) for results in outputs]
    outputs = tuple([np.asarray(x).reshape(broadcast_shape).squeeze() for x in zip(*outputs)])
    outputs = tuple([x.item() if np.ndim(x) == 0 else x for x in outputs])
    n_results = len(list(outputs))
    return outputs[0] if n_results == 1 else outputs