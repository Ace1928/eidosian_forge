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
def fz_write_header(self, w, h, n, alpha, xres, yres, pagenum, cs, seps):
    """
        Class-aware wrapper for `::fz_write_header()`.
        	Cause a band writer to write the header for
        	a banded image with the given properties/dimensions etc. This
        	also configures the bandwriter for the format of the data to be
        	passed in future calls.

        	w, h: Width and Height of the entire page.

        	n: Number of components (including spots and alphas).

        	alpha: Number of alpha components.

        	xres, yres: X and Y resolutions in dpi.

        	cs: Colorspace (NULL for bitmaps)

        	seps: Separation details (or NULL).
        """
    return _mupdf.FzBandWriter_fz_write_header(self, w, h, n, alpha, xres, yres, pagenum, cs, seps)