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
def fz_tree_archive_add_buffer(self, name, buf):
    """
        Class-aware wrapper for `::fz_tree_archive_add_buffer()`.
        	Add a named buffer to an existing tree archive.

        	The tree will take a new reference to the buffer. Ownership
        	is not transferred.
        """
    return _mupdf.FzArchive_fz_tree_archive_add_buffer(self, name, buf)