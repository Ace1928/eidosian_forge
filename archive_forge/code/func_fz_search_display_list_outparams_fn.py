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
def fz_search_display_list_outparams_fn(list, needle, hit_bbox, hit_max):
    """
    Class-aware helper for out-params of fz_search_display_list() [fz_search_display_list()].
    """
    ret, hit_mark = ll_fz_search_display_list(list.m_internal, needle, hit_bbox.internal(), hit_max)
    return (ret, hit_mark)