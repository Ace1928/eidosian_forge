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
def fz_decomp_image_from_stream_outparams_fn(stm, image, subarea, indexed, l2factor):
    """
    Class-aware helper for out-params of fz_decomp_image_from_stream() [fz_decomp_image_from_stream()].
    """
    ret, l2extra = ll_fz_decomp_image_from_stream(stm.m_internal, image.m_internal, subarea.internal(), indexed, l2factor)
    return (FzPixmap(ret), l2extra)