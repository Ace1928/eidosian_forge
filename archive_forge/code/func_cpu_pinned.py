import threading
import warnings
import ctypes
from .base import classproperty, with_metaclass, _MXClassPropertyMetaClass
from .base import _LIB
from .base import check_call
def cpu_pinned(device_id=0):
    """Returns a CPU pinned memory context. Copying from CPU pinned memory to GPU
    is faster than from normal CPU memory.

    This function is a short cut for ``Context('cpu_pinned', device_id)``.

    Examples
    ----------
    >>> with mx.cpu_pinned():
    ...     cpu_array = mx.nd.ones((2, 3))
    >>> cpu_array.context
    cpu_pinned(0)
    >>> cpu_array = mx.nd.ones((2, 3), ctx=mx.cpu_pinned())
    >>> cpu_array.context
    cpu_pinned(0)

    Parameters
    ----------
    device_id : int, optional
        The device id of the device. `device_id` is not needed for CPU.
        This is included to make interface compatible with GPU.

    Returns
    -------
    context : Context
        The corresponding CPU pinned memory context.
    """
    return Context('cpu_pinned', device_id)