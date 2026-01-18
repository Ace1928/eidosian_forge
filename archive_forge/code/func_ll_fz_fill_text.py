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
def ll_fz_fill_text(dev, text, ctm, colorspace, color, alpha, color_params):
    """
    Low-level Python version of fz_fill_text() taking list/tuple for `color`.
    """
    color = tuple(color) + (0,) * (4 - len(color))
    assert len(color) == 4, f'color not len 4: len={len(color)}: {color}'
    return ll_fz_fill_text2(dev, text, ctm, colorspace, *color, alpha, color_params)