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
def fz_open_archive(filename):
    """
    Class-aware wrapper for `::fz_open_archive()`.
    	Open a zip or tar archive

    	Open a file and identify its archive type based on the archive
    	signature contained inside.

    	filename: a path to a file as it would be given to open(2).
    """
    return _mupdf.fz_open_archive(filename)