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
def fz_pre_shear(self, sx, sy):
    """
        Class-aware wrapper for `::fz_pre_shear()`.
        	Premultiply a matrix with a shearing matrix.

        	The shearing matrix is of the form [ 1 sy sx 1 0 0 ].

        	m: pointer to matrix to premultiply

        	sx, sy: Shearing factors. A shearing factor of 0.0 will not
        	cause any shearing along the relevant axis.

        	Returns m (updated).
        """
    return _mupdf.FzMatrix_fz_pre_shear(self, sx, sy)