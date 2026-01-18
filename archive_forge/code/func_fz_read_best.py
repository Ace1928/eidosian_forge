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
def fz_read_best(self, initial, truncated, worst_case):
    """
        Class-aware wrapper for `::fz_read_best()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_read_best(size_t initial, size_t worst_case)` => `(fz_buffer *, int truncated)`

        	Attempt to read a stream into a buffer. If truncated
        	is NULL behaves as fz_read_all, sets a truncated flag in case of
        	error.

        	stm: The stream to read from.

        	initial: Suggested initial size for the buffer.

        	truncated: Flag to store success/failure indication in.

        	worst_case: 0 for unknown, otherwise an upper bound for the
        	size of the stream.

        	Returns a buffer created from reading from the stream.
        """
    return _mupdf.FzStream_fz_read_best(self, initial, truncated, worst_case)