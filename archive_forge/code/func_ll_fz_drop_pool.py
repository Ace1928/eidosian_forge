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
def ll_fz_drop_pool(pool):
    """
    Low-level wrapper for `::fz_drop_pool()`.
    Drop a pool, freeing and invalidating all storage returned from
    the pool.
    """
    return _mupdf.ll_fz_drop_pool(pool)