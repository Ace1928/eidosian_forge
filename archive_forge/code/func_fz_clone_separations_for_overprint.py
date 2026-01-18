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
def fz_clone_separations_for_overprint(self):
    """
        Class-aware wrapper for `::fz_clone_separations_for_overprint()`.
        	Return a separations object with all the spots in the input
        	separations object that are set to composite, reset to be
        	enabled. If there ARE no spots in the object, this returns
        	NULL. If the object already has all its spots enabled, then
        	just returns another handle on the same object.
        """
    return _mupdf.FzSeparations_fz_clone_separations_for_overprint(self)