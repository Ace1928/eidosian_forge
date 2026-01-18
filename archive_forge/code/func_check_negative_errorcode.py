import errno
import os
from ctypes import get_errno
def check_negative_errorcode(result, _func, *_args):
    """Error checker for funtions, which return negative error codes.

    If ``result`` is smaller than ``0``, it is interpreted as negative error
    code, and an appropriate exception is raised:

    - ``-ENOMEM`` raises a :exc:`~exceptions.MemoryError`
    - ``-EOVERFLOW`` raises a :exc:`~exceptions.OverflowError`
    - all other error codes raise :exc:`~exceptions.EnvironmentError`

    If result is greater or equal to ``0``, it is returned unchanged.

    """
    if result < 0:
        errnum = -result
        raise exception_from_errno(errnum)
    else:
        return result