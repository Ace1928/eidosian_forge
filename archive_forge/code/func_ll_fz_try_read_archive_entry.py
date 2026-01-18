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
def ll_fz_try_read_archive_entry(arch, name):
    """
    Low-level wrapper for `::fz_try_read_archive_entry()`.
    Reads all bytes in an archive entry
    into a buffer.

    name: Entry name to look for, this must be an exact match to
    the entry name in the archive.

    Returns NULL if a matching entry cannot be found. Otherwise behaves
    the same as fz_read_archive_entry. Exceptions may be thrown.
    """
    return _mupdf.ll_fz_try_read_archive_entry(arch, name)