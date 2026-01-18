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
def fz_include_point_in_rect(self, p):
    """
        Class-aware wrapper for `::fz_include_point_in_rect()`.
        	Expand a bbox to include a given point.
        	To create a rectangle that encompasses a sequence of points, the
        	rectangle must first be set to be the empty rectangle at one of
        	the points before including the others.
        """
    return _mupdf.FzRect_fz_include_point_in_rect(self, p)