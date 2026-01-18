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
def fz_compressed_image_type(self):
    """
        Class-aware wrapper for `::fz_compressed_image_type()`.
        	Return the type of a compressed image.

        	Any non-compressed image will have the type returned as UNKNOWN.
        """
    return _mupdf.FzImage_fz_compressed_image_type(self)