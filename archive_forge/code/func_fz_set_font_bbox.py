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
def fz_set_font_bbox(self, xmin, ymin, xmax, ymax):
    """
        Class-aware wrapper for `::fz_set_font_bbox()`.
        	Set the font bbox.

        	font: The font to set the bbox for.

        	xmin, ymin, xmax, ymax: The bounding box.
        """
    return _mupdf.FzFont_fz_set_font_bbox(self, xmin, ymin, xmax, ymax)