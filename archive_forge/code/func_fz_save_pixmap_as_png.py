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
def fz_save_pixmap_as_png(self, filename):
    """
        Class-aware wrapper for `::fz_save_pixmap_as_png()`.
        	Save a (Greyscale or RGB) pixmap as a png.
        """
    return _mupdf.FzPixmap_fz_save_pixmap_as_png(self, filename)