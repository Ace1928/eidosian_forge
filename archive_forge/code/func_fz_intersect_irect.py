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
def fz_intersect_irect(self, b):
    """
        Class-aware wrapper for `::fz_intersect_irect()`.
        	Compute intersection of two bounding boxes.

        	Similar to fz_intersect_rect but operates on two bounding
        	boxes instead of two rectangles.
        """
    return _mupdf.FzIrect_fz_intersect_irect(self, b)