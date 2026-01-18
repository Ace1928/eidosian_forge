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
def fz_list_archive_entry(self, idx):
    """
        Class-aware wrapper for `::fz_list_archive_entry()`.
        	Get listed name of entry position idx.

        	idx: Must be a value >= 0 < return value from
        	fz_count_archive_entries. If not in range NULL will be
        	returned.

        	May throw an exception if this type of archive cannot list the
        	entries (such as a directory).
        """
    return _mupdf.FzArchive_fz_list_archive_entry(self, idx)