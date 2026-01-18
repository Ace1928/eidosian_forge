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
def fz_transform_point(self, *args):
    """
        *Overload 1:*
         Class-aware wrapper for `::fz_transform_point()`.
        		Apply a transformation to a point.

        		transform: Transformation matrix to apply. See fz_concat,
        		fz_scale, fz_rotate and fz_translate for how to create a
        		matrix.

        		point: Pointer to point to update.

        		Returns transform (unchanged).


        |

        *Overload 2:*
         Class-aware wrapper for `::fz_transform_point()`.
        		Apply a transformation to a point.

        		transform: Transformation matrix to apply. See fz_concat,
        		fz_scale, fz_rotate and fz_translate for how to create a
        		matrix.

        		point: Pointer to point to update.

        		Returns transform (unchanged).
        """
    return _mupdf.FzPoint_fz_transform_point(self, *args)