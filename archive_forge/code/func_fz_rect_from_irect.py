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
def fz_rect_from_irect(self):
    """
        Class-aware wrapper for `::fz_rect_from_irect()`.
        	Convert a bbox into a rect.

        	For our purposes, a rect can represent all the values we meet in
        	a bbox, so nothing can go wrong.

        	rect: A place to store the generated rectangle.

        	bbox: The bbox to convert.

        	Returns rect (updated).
        """
    return _mupdf.FzIrect_fz_rect_from_irect(self)