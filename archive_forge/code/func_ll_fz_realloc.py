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
def ll_fz_realloc(p, size):
    """
    Low-level wrapper for `::fz_realloc()`.
    Reallocates a block of memory to given size. Existing contents
    up to min(old_size,new_size) are maintained. The rest of the
    block is uninitialised.

    fz_realloc(ctx, NULL, size) behaves like fz_malloc(ctx, size).

    fz_realloc(ctx, p, 0); behaves like fz_free(ctx, p).

    Throws exception in the event of failure to allocate.
    """
    return _mupdf.ll_fz_realloc(p, size)