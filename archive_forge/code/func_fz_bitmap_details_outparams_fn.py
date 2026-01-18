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
def fz_bitmap_details_outparams_fn(bitmap):
    """
    Class-aware helper for out-params of fz_bitmap_details() [fz_bitmap_details()].
    """
    w, h, n, stride = ll_fz_bitmap_details(bitmap.m_internal)
    return (w, h, n, stride)