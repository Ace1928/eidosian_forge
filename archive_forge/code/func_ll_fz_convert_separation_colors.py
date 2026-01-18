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
def ll_fz_convert_separation_colors(src_cs, src_color, dst_seps, dst_cs, color_params):
    """
    Wrapper for out-params of fz_convert_separation_colors().
    Returns: float dst_color
    """
    outparams = ll_fz_convert_separation_colors_outparams()
    ret = ll_fz_convert_separation_colors_outparams_fn(src_cs, src_color, dst_seps, dst_cs, color_params, outparams)
    return outparams.dst_color