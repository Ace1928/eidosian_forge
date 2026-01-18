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
def fz_process_opened_pages(self, process_openend_page, state):
    """
        Class-aware wrapper for `::fz_process_opened_pages()`.
        	Iterates over all opened pages of the document, calling the
        	provided callback for each page for processing. If the callback
        	returns non-NULL then the iteration stops and that value is returned
        	to the called of fz_process_opened_pages().

        	The state pointer provided to fz_process_opened_pages() is
        	passed on to the callback but is owned by the caller.

        	Returns the first non-NULL value returned by the callback,
        	or NULL if the callback returned NULL for all opened pages.
        """
    return _mupdf.FzDocument_fz_process_opened_pages(self, process_openend_page, state)