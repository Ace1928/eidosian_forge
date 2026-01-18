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
def fz_compress_ccitt_fax_g3(data, columns, rows, stride):
    """
    Class-aware wrapper for `::fz_compress_ccitt_fax_g3()`.
    	Compress bitmap data as CCITT Group 3 1D fax image.
    	Creates a stream assuming the default PDF parameters,
    	except the number of columns.
    """
    return _mupdf.fz_compress_ccitt_fax_g3(data, columns, rows, stride)