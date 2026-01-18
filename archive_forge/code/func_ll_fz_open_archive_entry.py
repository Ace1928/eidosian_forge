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
def ll_fz_open_archive_entry(arch, name):
    """
    Low-level wrapper for `::fz_open_archive_entry()`.
    Opens an archive entry as a stream.

    name: Entry name to look for, this must be an exact match to
    the entry name in the archive.

    Throws an exception if a matching entry cannot be found.
    """
    return _mupdf.ll_fz_open_archive_entry(arch, name)