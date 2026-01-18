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
def fz_is_quad_inside_quad(self, haystack):
    """
        Class-aware wrapper for `::fz_is_quad_inside_quad()`.
        	Inclusion test for quad in quad.

        	This may break down if quads are not 'well formed'.
        """
    return _mupdf.FzQuad_fz_is_quad_inside_quad(self, haystack)