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
def fz_outline_iterator_item(self):
    """
        Class-aware wrapper for `::fz_outline_iterator_item()`.
        	Call to get the current outline item.

        	Can return NULL. The item is only valid until the next call.
        """
    return _mupdf.FzOutlineIterator_fz_outline_iterator_item(self)