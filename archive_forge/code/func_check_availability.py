import functools as _functools
import numpy as _numpy
import platform as _platform
import cupy as _cupy
from cupy_backends.cuda.api import driver as _driver
from cupy_backends.cuda.api import runtime as _runtime
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupy._core import _dtype
from cupy.cuda import device as _device
from cupy.cuda import stream as _stream
from cupy import _util
import cupyx.scipy.sparse
@_util.memoize()
def check_availability(name):
    if not _runtime.is_hip:
        available_version = _available_cusparse_version
        version = _cusparse.get_build_version()
    else:
        available_version = _available_hipsparse_version
        version = _driver.get_build_version()
    if name not in available_version:
        msg = 'No available version information specified for {}'.format(name)
        raise ValueError(msg)
    version_added, version_removed = available_version[name]
    version_added = _get_version(version_added)
    version_removed = _get_version(version_removed)
    if version_added is not None and version < version_added:
        return False
    if version_removed is not None and version >= version_removed:
        return False
    return True