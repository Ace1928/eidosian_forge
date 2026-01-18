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
def fz_open_file(filename):
    """
    Class-aware wrapper for `::fz_open_file()`.
    	Open the named file and wrap it in a stream.

    	filename: Path to a file. On non-Windows machines the filename
    	should be exactly as it would be passed to fopen(2). On Windows
    	machines, the path should be UTF-8 encoded so that non-ASCII
    	characters can be represented. Other platforms do the encoding
    	as standard anyway (and in most cases, particularly for MacOS
    	and Linux, the encoding they use is UTF-8 anyway).
    """
    return _mupdf.fz_open_file(filename)