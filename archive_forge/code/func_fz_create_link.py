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
def fz_create_link(self, bbox, uri):
    """
        Class-aware wrapper for `::fz_create_link()`.
        	Create a new link on a page.
        """
    return _mupdf.FzPage_fz_create_link(self, bbox, uri)