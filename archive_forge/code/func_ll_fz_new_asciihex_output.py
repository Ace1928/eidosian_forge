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
def ll_fz_new_asciihex_output(chain):
    """
    Low-level wrapper for `::fz_new_asciihex_output()`.
    Compression and other filtering outputs.

    These outputs write encoded data to another output. Create a
    filter output with the destination, write to the filter, then
    close and drop it when you're done. These can also be chained
    together, for example to write ASCII Hex encoded, Deflate
    compressed, and RC4 encrypted data to a buffer output.

    Output streams don't use reference counting, so make sure to
    close all of the filters in the reverse order of creation so
    that data is flushed properly.

    Accordingly, ownership of 'chain' is never passed into the
    following functions, but remains with the caller, whose
    responsibility it is to ensure they exist at least until
    the returned fz_output is dropped.
    """
    return _mupdf.ll_fz_new_asciihex_output(chain)