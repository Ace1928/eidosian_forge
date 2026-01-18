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
def fz_unshare_stroke_state(self):
    """
        Class-aware wrapper for `::fz_unshare_stroke_state()`.
        	Given a reference to a (possibly) shared stroke_state structure,
        	return a reference to an equivalent stroke_state structure
        	that is guaranteed to be unshared (i.e. one that can
        	safely be modified).

        	shared: The reference to a (possibly) shared structure
        	to unshare. Ownership of this reference is passed in
        	to this function, even in the case of exceptions being
        	thrown.

        	Exceptions may be thrown in the event of failure to
        	allocate if required.
        """
    return _mupdf.FzStrokeState_fz_unshare_stroke_state(self)