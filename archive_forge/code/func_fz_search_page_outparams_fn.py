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
def fz_search_page_outparams_fn(page, needle, hit_bbox, hit_max):
    """
    Class-aware helper for out-params of fz_search_page() [fz_search_page()].
    """
    ret, hit_mark = ll_fz_search_page(page.m_internal, needle, hit_bbox.internal(), hit_max)
    return (ret, hit_mark)