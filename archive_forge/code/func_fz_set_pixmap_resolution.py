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
def fz_set_pixmap_resolution(self, xres, yres):
    """
        Class-aware wrapper for `::fz_set_pixmap_resolution()`.
        	Set the pixels per inch resolution of the pixmap.
        """
    return _mupdf.FzPixmap_fz_set_pixmap_resolution(self, xres, yres)