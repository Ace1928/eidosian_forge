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
def ll_fz_highlight_selection2(page, a, b, max_quads):
    """
     Low-level wrapper for `::fz_highlight_selection2()`.
    C++ alternative to fz_highlight_selection() that returns quads in a
    std::vector.
    """
    return _mupdf.ll_fz_highlight_selection2(page, a, b, max_quads)