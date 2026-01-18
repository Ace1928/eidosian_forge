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
def fz_read_line(self, buf, max):
    """
        Class-aware wrapper for `::fz_read_line()`.
        	Read a line from stream into the buffer until either a
        	terminating newline or EOF, which it replaces with a null byte
        	('').

        	Returns buf on success, and NULL when end of file occurs while
        	no characters have been read.
        """
    return _mupdf.FzStream_fz_read_line(self, buf, max)