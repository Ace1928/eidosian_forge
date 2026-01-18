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
def fz_page_separations(self):
    """
        Class-aware wrapper for `::fz_page_separations()`.
        	Get the separations details for a page.
        	This will be NULL, unless the format specifically supports
        	separations (such as PDF files). May be NULL even
        	so, if there are no separations on a page.

        	Returns a reference that must be dropped.
        """
    return _mupdf.FzPage_fz_page_separations(self)