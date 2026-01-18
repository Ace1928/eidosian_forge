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
def ll_fz_pool_size(pool):
    """
    Low-level wrapper for `::fz_pool_size()`.
    The current size of the pool.

    The number of bytes of storage currently allocated to the pool.
    This is the total of the storage used for the blocks making
    up the pool, rather then total of the allocated blocks so far,
    so it will increase in 'lumps'.
    from the pool, then the pool size may still be X
    """
    return _mupdf.ll_fz_pool_size(pool)