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
def fz_delete_link(self, link):
    """
        Class-aware wrapper for `::fz_delete_link()`.
        	Delete an existing link on a page.
        """
    return _mupdf.FzPage_fz_delete_link(self, link)