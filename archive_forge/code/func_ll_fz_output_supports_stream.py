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
def ll_fz_output_supports_stream(out):
    """
    Low-level wrapper for `::fz_output_supports_stream()`.
    Query whether a given fz_output supports fz_stream_from_output.
    """
    return _mupdf.ll_fz_output_supports_stream(out)