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
def fz_new_page_of_size(size, doc):
    """
    Class-aware wrapper for `::fz_new_page_of_size()`.
    	Different document types will be implemented by deriving from
    	fz_page. This macro allocates such derived structures, and
    	initialises the base sections.
    """
    return _mupdf.fz_new_page_of_size(size, doc)