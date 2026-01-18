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
def ll_fz_lookup_cjk_font_by_language(lang):
    """
    Wrapper for out-params of fz_lookup_cjk_font_by_language().
    Returns: const unsigned char *, int len, int subfont
    """
    outparams = ll_fz_lookup_cjk_font_by_language_outparams()
    ret = ll_fz_lookup_cjk_font_by_language_outparams_fn(lang, outparams)
    return (ret, outparams.len, outparams.subfont)