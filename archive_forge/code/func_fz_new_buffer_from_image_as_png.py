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
def fz_new_buffer_from_image_as_png(self, color_params):
    """
        Class-aware wrapper for `::fz_new_buffer_from_image_as_png()`.
        	Reencode a given image as a PNG into a buffer.

        	Ownership of the buffer is returned.
        """
    return _mupdf.FzImage_fz_new_buffer_from_image_as_png(self, color_params)