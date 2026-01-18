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
def fz_open_dctd(self, color_transform, invert_cmyk, l2factor, jpegtables):
    """
        Class-aware wrapper for `::fz_open_dctd()`.
        	dctd filter performs DCT (JPEG) decoding of data read
        	from the chained filter.

        	color_transform implements the PDF color_transform option
        		use -1 for default behavior
        		use 0 to disable YUV-RGB / YCCK-CMYK transforms
        		use 1 to enable YUV-RGB / YCCK-CMYK transforms

        	invert_cmyk implements the necessary inversion for Photoshop CMYK images
        		use 0 if embedded in PDF
        		use 1 if not embedded in PDF

        	For subsampling on decode, set l2factor to the log2 of the
        	reduction required (therefore 0 = full size decode).

        	jpegtables is an optional stream from which the JPEG tables
        	can be read. Use NULL if not required.
        """
    return _mupdf.FzStream_fz_open_dctd(self, color_transform, invert_cmyk, l2factor, jpegtables)