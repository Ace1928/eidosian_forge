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
def fz_image_resolution(self, xres, yres):
    """
        Class-aware wrapper for `::fz_image_resolution()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_image_resolution()` => `(int xres, int yres)`

        	Request the natural resolution
        	of an image.

        	xres, yres: Pointers to ints to be updated with the
        	natural resolution of an image (or a sensible default
        	if not encoded).
        """
    return _mupdf.FzImage_fz_image_resolution(self, xres, yres)