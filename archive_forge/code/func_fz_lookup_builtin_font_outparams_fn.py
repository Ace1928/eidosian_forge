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
def fz_lookup_builtin_font_outparams_fn(name, bold, italic):
    """
    Class-aware helper for out-params of fz_lookup_builtin_font() [fz_lookup_builtin_font()].
    """
    ret, len = ll_fz_lookup_builtin_font(name, bold, italic)
    return (ret, len)