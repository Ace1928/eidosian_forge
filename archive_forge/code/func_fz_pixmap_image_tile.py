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
def fz_pixmap_image_tile(self):
    """
        Class-aware wrapper for `::fz_pixmap_image_tile()`.
        	Retrieve the underlying fz_pixmap for an image.

        	Returns a pointer to the underlying fz_pixmap for an image,
        	or NULL if this image is not based upon an fz_pixmap.

        	No reference is returned. Lifespan is limited to that of
        	the image itself. If required, use fz_keep_pixmap to take
        	a reference to keep it longer.
        """
    return _mupdf.FzPixmapImage_fz_pixmap_image_tile(self)