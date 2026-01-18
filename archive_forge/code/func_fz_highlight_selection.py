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
def fz_highlight_selection(self, a, b, quads, max_quads):
    """
        Class-aware wrapper for `::fz_highlight_selection()`.
        	Return a list of quads to highlight lines inside the selection
        	points.
        """
    return _mupdf.FzStextPage_fz_highlight_selection(self, a, b, quads, max_quads)