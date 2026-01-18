import functools
import gc
import inspect
import torch
from .imports import is_mps_available, is_npu_available, is_xpu_available
def release_memory(*objects):
    """
    Releases memory from `objects` by setting them to `None` and calls `gc.collect()` and `torch.cuda.empty_cache()`.
    Returned objects should be reassigned to the same variables.

    Args:
        objects (`Iterable`):
            An iterable of objects
    Returns:
        A list of `None` objects to replace `objects`

    Example:

        ```python
        >>> import torch
        >>> from accelerate.utils import release_memory

        >>> a = torch.ones(1000, 1000).cuda()
        >>> b = torch.ones(1000, 1000).cuda()
        >>> a, b = release_memory(a, b)
        ```
    """
    if not isinstance(objects, list):
        objects = list(objects)
    for i in range(len(objects)):
        objects[i] = None
    gc.collect()
    if is_xpu_available():
        torch.xpu.empty_cache()
    elif is_npu_available():
        torch.npu.empty_cache()
    elif is_mps_available():
        torch.mps.empty_cache()
    else:
        torch.cuda.empty_cache()
    return objects