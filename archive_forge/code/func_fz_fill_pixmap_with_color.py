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
def fz_fill_pixmap_with_color(self, colorspace, color, color_params):
    """
        Class-aware wrapper for `::fz_fill_pixmap_with_color()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_fill_pixmap_with_color(::fz_colorspace *colorspace, ::fz_color_params color_params)` => float color

        	Fill pixmap with solid color.
        """
    return _mupdf.FzPixmap_fz_fill_pixmap_with_color(self, colorspace, color, color_params)