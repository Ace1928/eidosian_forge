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
def fz_open_memory(data, len):
    """
    Class-aware wrapper for `::fz_open_memory()`.
    	Open a block of memory as a stream.

    	data: Pointer to start of data block. Ownership of the data
    	block is NOT passed in.

    	len: Number of bytes in data block.

    	Returns pointer to newly created stream. May throw exceptions on
    	failure to allocate.
    """
    return _mupdf.fz_open_memory(data, len)