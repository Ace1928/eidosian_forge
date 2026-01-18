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
def fz_dom_append_child(self, child):
    """
        Class-aware wrapper for `::fz_dom_append_child()`.
        	Insert an element as the last child of a parent, unlinking the
        	child from its current position if required.
        """
    return _mupdf.FzXml_fz_dom_append_child(self, child)