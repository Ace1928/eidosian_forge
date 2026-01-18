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
def ll_fz_set_font_embedding(font, embed):
    """
    Low-level wrapper for `::fz_set_font_embedding()`.
    Control whether a given font should be embedded or not when writing.
    """
    return _mupdf.ll_fz_set_font_embedding(font, embed)