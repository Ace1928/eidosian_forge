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
def fz_open_image_decomp_stream(self, arg_1, l2factor):
    """
        Class-aware wrapper for `::fz_open_image_decomp_stream()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_open_image_decomp_stream(::fz_compression_params *arg_1)` => `(fz_stream *, int l2factor)`

        	Open a stream to read the decompressed version of another stream
        	with optional log2 subsampling.
        """
    return _mupdf.FzStream_fz_open_image_decomp_stream(self, arg_1, l2factor)