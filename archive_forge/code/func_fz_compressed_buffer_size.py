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
def fz_compressed_buffer_size(self):
    """
        Class-aware wrapper for `::fz_compressed_buffer_size()`.
        	Return the storage size used for a buffer and its data.
        	Used in implementing store handling.

        	Never throws exceptions.
        """
    return _mupdf.FzCompressedBuffer_fz_compressed_buffer_size(self)