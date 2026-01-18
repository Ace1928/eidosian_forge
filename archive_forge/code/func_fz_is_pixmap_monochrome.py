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
def fz_is_pixmap_monochrome(self):
    """
        Class-aware wrapper for `::fz_is_pixmap_monochrome()`.
        	Check if the pixmap is a 1-channel image containing samples with
        	only values 0 and 255
        """
    return _mupdf.FzPixmap_fz_is_pixmap_monochrome(self)