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
def fz_bound_path(self, stroke, ctm):
    """
        Class-aware wrapper for `::fz_bound_path()`.
        	Return a bounding rectangle for a path.

        	path: The path to bound.

        	stroke: If NULL, the bounding rectangle given is for
        	the filled path. If non-NULL the bounding rectangle
        	given is for the path stroked with the given attributes.

        	ctm: The matrix to apply to the path during stroking.

        	r: Pointer to a fz_rect which will be used to hold
        	the result.

        	Returns r, updated to contain the bounding rectangle.
        """
    return _mupdf.FzPath_fz_bound_path(self, stroke, ctm)