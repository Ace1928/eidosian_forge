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
def fz_dom_insert_after(self, new_elt):
    """
        Class-aware wrapper for `::fz_dom_insert_after()`.
        	Insert an element (new_elt), after another element (node),
        	unlinking the new_elt from its current position if required.
        """
    return _mupdf.FzXml_fz_dom_insert_after(self, new_elt)