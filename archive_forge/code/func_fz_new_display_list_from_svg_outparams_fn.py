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
def fz_new_display_list_from_svg_outparams_fn(buf, base_uri, dir):
    """
    Class-aware helper for out-params of fz_new_display_list_from_svg() [fz_new_display_list_from_svg()].
    """
    ret, w, h = ll_fz_new_display_list_from_svg(buf.m_internal, base_uri, dir.m_internal)
    return (FzDisplayList(ret), w, h)