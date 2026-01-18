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
def fz_new_image_from_pixmap(self, mask):
    """
        Class-aware wrapper for `::fz_new_image_from_pixmap()`.
        	Create an image from the given
        	pixmap.

        	pixmap: The pixmap to base the image upon. A new reference
        	to this is taken.

        	mask: NULL, or another image to use as a mask for this one.
        	A new reference is taken to this image. Supplying a masked
        	image as a mask to another image is illegal!
        """
    return _mupdf.FzPixmap_fz_new_image_from_pixmap(self, mask)