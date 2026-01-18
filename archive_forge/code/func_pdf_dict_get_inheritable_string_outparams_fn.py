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
def pdf_dict_get_inheritable_string_outparams_fn(dict, key):
    """
    Class-aware helper for out-params of pdf_dict_get_inheritable_string() [pdf_dict_get_inheritable_string()].
    """
    ret, sizep = ll_pdf_dict_get_inheritable_string(dict.m_internal, key.m_internal)
    return (ret, sizep)