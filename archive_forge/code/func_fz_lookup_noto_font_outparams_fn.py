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
def fz_lookup_noto_font_outparams_fn(script, lang):
    """
    Class-aware helper for out-params of fz_lookup_noto_font() [fz_lookup_noto_font()].
    """
    ret, len, subfont = ll_fz_lookup_noto_font(script, lang)
    return (ret, len, subfont)