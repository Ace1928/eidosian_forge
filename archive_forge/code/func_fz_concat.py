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
def fz_concat(self, *args):
    """
        *Overload 1:*
         We use default copy constructor and operator=.  Class-aware wrapper for `::fz_concat()`.
        		Multiply two matrices.

        		The order of the two matrices are important since matrix
        		multiplication is not commutative.

        		Returns result.


        |

        *Overload 2:*
         Class-aware wrapper for `::fz_concat()`.
        		Multiply two matrices.

        		The order of the two matrices are important since matrix
        		multiplication is not commutative.

        		Returns result.
        """
    return _mupdf.FzMatrix_fz_concat(self, *args)