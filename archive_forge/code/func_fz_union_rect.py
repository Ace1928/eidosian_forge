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
def fz_union_rect(self, *args):
    """
        *Overload 1:*
         Class-aware wrapper for `::fz_union_rect()`.
        		Compute union of two rectangles.

        		Given two rectangles, update the first to be the smallest
        		axis-aligned rectangle that encompasses both given rectangles.
        		If either rectangle is infinite then the union is also infinite.
        		If either rectangle is empty then the union is simply the
        		non-empty rectangle. Should both rectangles be empty, then the
        		union is also empty.


        |

        *Overload 2:*
         Class-aware wrapper for `::fz_union_rect()`.
        		Compute union of two rectangles.

        		Given two rectangles, update the first to be the smallest
        		axis-aligned rectangle that encompasses both given rectangles.
        		If either rectangle is infinite then the union is also infinite.
        		If either rectangle is empty then the union is simply the
        		non-empty rectangle. Should both rectangles be empty, then the
        		union is also empty.
        """
    return _mupdf.FzRect_fz_union_rect(self, *args)