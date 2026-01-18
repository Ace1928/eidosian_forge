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
def fz_parse_page_range_outparams_fn(s, n):
    """
    Class-aware helper for out-params of fz_parse_page_range() [fz_parse_page_range()].
    """
    ret, a, b = ll_fz_parse_page_range(s, n)
    return (ret, a, b)