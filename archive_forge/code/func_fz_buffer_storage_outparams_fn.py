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
def fz_buffer_storage_outparams_fn(buf):
    """
    Class-aware helper for out-params of fz_buffer_storage() [fz_buffer_storage()].
    """
    ret, datap = ll_fz_buffer_storage(buf.m_internal)
    return (ret, datap)