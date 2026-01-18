from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
def ll_fz_drop_device(dev):
    """
    Low-level wrapper for `::fz_drop_device()`.
    Reduce the reference count on a device. When the reference count
    reaches zero, the device and its resources will be freed.
    Don't forget to call fz_close_device before dropping the device,
    or you may get incomplete output!

    Never throws exceptions.
    """
    return _mupdf.ll_fz_drop_device(dev)