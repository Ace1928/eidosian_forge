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
def fz_convert_separation_colors_outparams_fn(src_cs, src_color, dst_seps, dst_cs, color_params):
    """
    Class-aware helper for out-params of fz_convert_separation_colors() [fz_convert_separation_colors()].
    """
    dst_color = ll_fz_convert_separation_colors(src_cs.m_internal, src_color, dst_seps.m_internal, dst_cs.m_internal, color_params.internal())
    return dst_color