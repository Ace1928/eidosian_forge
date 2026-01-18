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
def fz_buffer_storage(self, datap):
    """
        Class-aware wrapper for `::fz_buffer_storage()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_buffer_storage()` => `(size_t, unsigned char *datap)`

        	Retrieve internal memory of buffer.

        	datap: Output parameter that will be pointed to the data.

        	Returns the current size of the data in bytes.
        """
    return _mupdf.FzBuffer_fz_buffer_storage(self, datap)