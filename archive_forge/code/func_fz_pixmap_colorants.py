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
def fz_pixmap_colorants(self):
    """
        Class-aware wrapper for `::fz_pixmap_colorants()`.
        	Return the number of colorants in a pixmap.

        	Returns the number of colorants (components, less any spots and
        	alpha).
        """
    return _mupdf.FzPixmap_fz_pixmap_colorants(self)