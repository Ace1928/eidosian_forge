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
def ll_fz_new_store_context(max):
    """
    Low-level wrapper for `::fz_new_store_context()`.
    Create a new store inside the context

    max: The maximum size (in bytes) that the store is allowed to
    grow to. FZ_STORE_UNLIMITED means no limit.
    """
    return _mupdf.ll_fz_new_store_context(max)