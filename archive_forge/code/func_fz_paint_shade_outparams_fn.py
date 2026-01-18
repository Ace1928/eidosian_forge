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
def fz_paint_shade_outparams_fn(shade, override_cs, ctm, dest, color_params, bbox, eop):
    """
    Class-aware helper for out-params of fz_paint_shade() [fz_paint_shade()].
    """
    cache = ll_fz_paint_shade(shade.m_internal, override_cs.m_internal, ctm.internal(), dest.m_internal, color_params.internal(), bbox.internal(), eop.m_internal)
    return FzShadeColorCache(ll_fz_keep_shade_color_cache(cache))