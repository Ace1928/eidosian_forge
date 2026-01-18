import functools
import gc
import inspect
import torch
from .imports import is_mps_available, is_npu_available, is_xpu_available
def find_executable_batch_size(function: callable=None, starting_batch_size: int=128):
    """
    A basic decorator that will try to execute `function`. If it fails from exceptions related to out-of-memory or
    CUDNN, the batch size is cut in half and passed to `function`

    `function` must take in a `batch_size` parameter as its first argument.

    Args:
        function (`callable`, *optional*):
            A function to wrap
        starting_batch_size (`int`, *optional*):
            The batch size to try and fit into memory

    Example:

    ```python
    >>> from accelerate.utils import find_executable_batch_size


    >>> @find_executable_batch_size(starting_batch_size=128)
    ... def train(batch_size, model, optimizer):
    ...     ...


    >>> train(model, optimizer)
    ```
    """
    if function is None:
        return functools.partial(find_executable_batch_size, starting_batch_size=starting_batch_size)
    batch_size = starting_batch_size

    def decorator(*args, **kwargs):
        nonlocal batch_size
        gc.collect()
        if is_xpu_available():
            torch.xpu.empty_cache()
        elif is_npu_available():
            torch.npu.empty_cache()
        else:
            torch.cuda.empty_cache()
        params = list(inspect.signature(function).parameters.keys())
        if len(params) < len(args) + 1:
            arg_str = ', '.join([f'{arg}={value}' for arg, value in zip(params[1:], args[1:])])
            raise TypeError(f'Batch size was passed into `{function.__name__}` as the first argument when called.Remove this as the decorator already does so: `{function.__name__}({arg_str})`')
        while True:
            if batch_size == 0:
                raise RuntimeError('No executable batch size found, reached zero.')
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    gc.collect()
                    if is_xpu_available():
                        torch.xpu.empty_cache()
                    elif is_npu_available():
                        torch.npu.empty_cache()
                    else:
                        torch.cuda.empty_cache()
                    batch_size //= 2
                else:
                    raise
    return decorator