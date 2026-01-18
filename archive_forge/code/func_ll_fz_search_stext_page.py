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
def ll_fz_search_stext_page(text, needle, hit_bbox, hit_max):
    """
    Wrapper for out-params of fz_search_stext_page().
    Returns: int, int hit_mark
    """
    outparams = ll_fz_search_stext_page_outparams()
    ret = ll_fz_search_stext_page_outparams_fn(text, needle, hit_bbox, hit_max, outparams)
    return (ret, outparams.hit_mark)