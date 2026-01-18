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
def fz_translate_rect(self, xoff, yoff):
    """
        Class-aware wrapper for `::fz_translate_rect()`.
        	Translate bounding box.

        	Translate a bbox by a given x and y offset. Allows for overflow.
        """
    return _mupdf.FzRect_fz_translate_rect(self, xoff, yoff)