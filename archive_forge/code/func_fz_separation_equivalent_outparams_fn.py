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
def fz_separation_equivalent_outparams_fn(seps, idx, dst_cs, prf, color_params):
    """
    Class-aware helper for out-params of fz_separation_equivalent() [fz_separation_equivalent()].
    """
    dst_color = ll_fz_separation_equivalent(seps.m_internal, idx, dst_cs.m_internal, prf.m_internal, color_params.internal())
    return dst_color