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
def ll_fz_append_data(buf, data, len):
    """
    Low-level wrapper for `::fz_append_data()`.
    fz_append_*: Append data to a buffer.

    The buffer will automatically grow as required.
    """
    return _mupdf.ll_fz_append_data(buf, data, len)