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
def fz_get_unscaled_pixmap_from_image(self):
    """
        Class-aware wrapper for `::fz_get_unscaled_pixmap_from_image()`.
        	Calls fz_get_pixmap_from_image() with ctm, subarea, w and h all set to NULL.
        """
    return _mupdf.FzImage_fz_get_unscaled_pixmap_from_image(self)