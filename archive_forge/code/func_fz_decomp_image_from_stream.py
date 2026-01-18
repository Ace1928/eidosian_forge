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
def fz_decomp_image_from_stream(self, image, subarea, indexed, l2factor, l2extra):
    """
        Class-aware wrapper for `::fz_decomp_image_from_stream()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_decomp_image_from_stream(::fz_compressed_image *image, ::fz_irect *subarea, int indexed, int l2factor)` => `(fz_pixmap *, int l2extra)`

        	Decode a subarea of a compressed image. l2factor is the amount
        	of subsampling inbuilt to the stream (i.e. performed by the
        	decoder). If non NULL, l2extra is the extra amount of
        	subsampling that should be performed by this routine. This will
        	be updated on exit to the amount of subsampling that is still
        	required to be done.

        	Returns a kept reference.
        """
    return _mupdf.FzStream_fz_decomp_image_from_stream(self, image, subarea, indexed, l2factor, l2extra)