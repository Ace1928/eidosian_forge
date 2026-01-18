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
def fz_clear_pixmap(self):
    """
        Class-aware wrapper for `::fz_clear_pixmap()`.
        	Sets all components (including alpha) of
        	all pixels in a pixmap to 0.

        	pix: The pixmap to clear.
        """
    return _mupdf.FzPixmap_fz_clear_pixmap(self)