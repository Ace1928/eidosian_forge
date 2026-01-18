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
def ll_fz_separation_equivalent(seps, idx, dst_cs, prf, color_params):
    """
    Wrapper for out-params of fz_separation_equivalent().
    Returns: float dst_color
    """
    outparams = ll_fz_separation_equivalent_outparams()
    ret = ll_fz_separation_equivalent_outparams_fn(seps, idx, dst_cs, prf, color_params, outparams)
    return outparams.dst_color