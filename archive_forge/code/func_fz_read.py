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
def fz_read(self, data, len):
    """
        Class-aware wrapper for `::fz_read()`.
        	Read from a stream into a given data block.

        	stm: The stream to read from.

        	data: The data block to read into.

        	len: The length of the data block (in bytes).

        	Returns the number of bytes read. May throw exceptions.
        """
    return _mupdf.FzStream_fz_read(self, data, len)