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
def fz_pre_rotate(self, degrees):
    """
        Class-aware wrapper for `::fz_pre_rotate()`.
        	Rotate a transformation by premultiplying.

        	The premultiplied matrix is of the form
        	[ cos(deg) sin(deg) -sin(deg) cos(deg) 0 0 ].

        	m: Pointer to matrix to premultiply.

        	degrees: Degrees of counter clockwise rotation. Values less
        	than zero and greater than 360 are handled as expected.

        	Returns m (updated).
        """
    return _mupdf.FzMatrix_fz_pre_rotate(self, degrees)