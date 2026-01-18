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
def ll_fz_new_draw_device_with_options(options, mediabox):
    """
    Wrapper for out-params of fz_new_draw_device_with_options().
    Returns: fz_device *, ::fz_pixmap *pixmap
    """
    outparams = ll_fz_new_draw_device_with_options_outparams()
    ret = ll_fz_new_draw_device_with_options_outparams_fn(options, mediabox, outparams)
    return (ret, outparams.pixmap)