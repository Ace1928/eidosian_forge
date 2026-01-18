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
def ll_fz_is_page_range(s):
    """
     Low-level wrapper for `::fz_is_page_range()`.
    	Check and parse string into page ranges:
    ,?(-?+|N)(-(-?+|N))?/
    """
    return _mupdf.ll_fz_is_page_range(s)