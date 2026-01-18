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
def ll_fz_matrix_max_expansion(m):
    """
    Low-level wrapper for `::fz_matrix_max_expansion()`.
    Find the largest expansion performed by this matrix.
    (i.e. max(abs(m.a),abs(m.b),abs(m.c),abs(m.d))
    """
    return _mupdf.ll_fz_matrix_max_expansion(m)