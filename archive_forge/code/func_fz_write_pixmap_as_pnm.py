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
def fz_write_pixmap_as_pnm(self, pixmap):
    """
        Class-aware wrapper for `::fz_write_pixmap_as_pnm()`.
        	Write a pixmap as a pnm (greyscale or rgb, no alpha).
        """
    return _mupdf.FzOutput_fz_write_pixmap_as_pnm(self, pixmap)