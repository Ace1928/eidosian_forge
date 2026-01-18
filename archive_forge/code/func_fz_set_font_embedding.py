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
def fz_set_font_embedding(self, embed):
    """
        Class-aware wrapper for `::fz_set_font_embedding()`.
        	Control whether a given font should be embedded or not when writing.
        """
    return _mupdf.FzFont_fz_set_font_embedding(self, embed)