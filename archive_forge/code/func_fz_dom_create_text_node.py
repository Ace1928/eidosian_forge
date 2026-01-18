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
def fz_dom_create_text_node(self, text):
    """
        Class-aware wrapper for `::fz_dom_create_text_node()`.
        	Create a text node for the given DOM.

        	The element is not linked into the DOM yet.
        """
    return _mupdf.FzXml_fz_dom_create_text_node(self, text)