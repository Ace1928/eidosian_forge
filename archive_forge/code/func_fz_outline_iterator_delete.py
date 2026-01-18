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
def fz_outline_iterator_delete(self):
    """
        Class-aware wrapper for `::fz_outline_iterator_delete()`.
        	Delete the current item.

        	This implicitly moves us to the 'next' item, and the return code is as for fz_outline_iterator_next.
        """
    return _mupdf.FzOutlineIterator_fz_outline_iterator_delete(self)