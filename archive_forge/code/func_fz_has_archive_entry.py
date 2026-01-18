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
def fz_has_archive_entry(self, name):
    """
        Class-aware wrapper for `::fz_has_archive_entry()`.
        	Check if entry by given name exists.

        	If named entry does not exist 0 will be returned, if it does
        	exist 1 is returned.

        	name: Entry name to look for, this must be an exact match to
        	the entry name in the archive.
        """
    return _mupdf.FzArchive_fz_has_archive_entry(self, name)