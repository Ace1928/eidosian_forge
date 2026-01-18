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
def fz_seek(self, offset, whence):
    """
        Class-aware wrapper for `::fz_seek()`.
        	Seek within a stream.

        	stm: The stream to seek within.

        	offset: The offset to seek to.

        	whence: From where the offset is measured (see fseek).
        	SEEK_SET - start of stream.
        	SEEK_CUR - current position.
        	SEEK_END - end of stream.

        """
    return _mupdf.FzStream_fz_seek(self, offset, whence)