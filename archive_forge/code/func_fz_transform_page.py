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
def fz_transform_page(self, resolution, rotate):
    """
        Class-aware wrapper for `::fz_transform_page()`.
        	Create transform matrix to draw page
        	at a given resolution and rotation. Adjusts the scaling
        	factors so that the page covers whole number of
        	pixels and adjust the page origin to be at 0,0.
        """
    return _mupdf.FzRect_fz_transform_page(self, resolution, rotate)