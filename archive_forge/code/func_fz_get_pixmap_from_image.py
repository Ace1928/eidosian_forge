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
def fz_get_pixmap_from_image(self, subarea, ctm, w, h):
    """
        Class-aware wrapper for `::fz_get_pixmap_from_image()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_get_pixmap_from_image(const ::fz_irect *subarea, ::fz_matrix *ctm)` => `(fz_pixmap *, int w, int h)`

        	Called to get a handle to a pixmap from an image.

        	image: The image to retrieve a pixmap from.

        	subarea: The subarea of the image that we actually care about
        	(or NULL to indicate the whole image).

        	ctm: Optional, unless subarea is given. If given, then on
        	entry this is the transform that will be applied to the complete
        	image. It should be updated on exit to the transform to apply to
        	the given subarea of the image. This is used to calculate the
        	desired width/height for subsampling.

        	w: If non-NULL, a pointer to an int to be updated on exit to the
        	width (in pixels) that the scaled output will cover.

        	h: If non-NULL, a pointer to an int to be updated on exit to the
        	height (in pixels) that the scaled output will cover.

        	Returns a non NULL kept pixmap pointer. May throw exceptions.
        """
    return _mupdf.FzImage_fz_get_pixmap_from_image(self, subarea, ctm, w, h)