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
def fz_dom_first_child(self):
    """
        Class-aware wrapper for `::fz_dom_first_child()`.
        	Return a borrowed reference to the first child of a node,
        	or NULL if there isn't one.
        """
    return _mupdf.FzXml_fz_dom_first_child(self)