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
def fz_buffer_extract_outparams_fn(buf):
    """
    Class-aware helper for out-params of fz_buffer_extract() [fz_buffer_extract()].
    """
    ret, data = ll_fz_buffer_extract(buf.m_internal)
    return (ret, data)