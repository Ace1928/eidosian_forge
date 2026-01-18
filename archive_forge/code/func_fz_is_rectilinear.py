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
def fz_is_rectilinear(self):
    """
        Class-aware wrapper for `::fz_is_rectilinear()`.
        	Check if a transformation is rectilinear.

        	Rectilinear means that no shearing is present and that any
        	rotations present are a multiple of 90 degrees. Usually this
        	is used to make sure that axis-aligned rectangles before the
        	transformation are still axis-aligned rectangles afterwards.
        """
    return _mupdf.FzMatrix_fz_is_rectilinear(self)