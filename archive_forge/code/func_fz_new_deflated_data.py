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
def fz_new_deflated_data(compressed_length, source, source_length, level):
    """
     Class-aware wrapper for `::fz_new_deflated_data()`.

    	This function has out-params. Python/C# wrappers look like:
    		`fz_new_deflated_data(const unsigned char *source, size_t source_length, ::fz_deflate_level level)` => `(unsigned char *, size_t compressed_length)`

    		Compress source_length bytes of data starting
    		at source, into a new memory block malloced for that purpose.
    compressed_length is updated on exit to contain the size used.
    		Ownership of the block is returned from this function, and the
    		caller is therefore responsible for freeing it. The block may be
    		considerably larger than is actually required. The caller is
    		free to fz_realloc it down if it wants to.
    """
    return _mupdf.fz_new_deflated_data(compressed_length, source, source_length, level)