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
@staticmethod
def fz_translate(tx, ty):
    """
        Class-aware wrapper for `::fz_translate()`.
        	Create a translation matrix.

        	The returned matrix is of the form [ 1 0 0 1 tx ty ].

        	m: A place to store the created matrix.

        	tx, ty: Translation distances along the X- and Y-axes. A
        	translation of 0 will not cause any translation along the
        	relevant axis.

        	Returns m.
        """
    return _mupdf.FzMatrix_fz_translate(tx, ty)