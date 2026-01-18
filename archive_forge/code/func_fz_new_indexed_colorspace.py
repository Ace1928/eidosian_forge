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
def fz_new_indexed_colorspace(self, high, lookup):
    """
        Class-aware wrapper for `::fz_new_indexed_colorspace()`.
        	Create an indexed colorspace.

        	The supplied lookup table is high palette entries long. Each
        	entry is n bytes long, where n is given by the number of
        	colorants in the base colorspace, one byte per colorant.

        	Ownership of lookup is passed it; it will be freed on
        	destruction, so must be heap allocated.

        	The colorspace will keep an additional reference to the base
        	colorspace that will be dropped on destruction.

        	The returned reference should be dropped when it is finished
        	with.

        	Colorspaces are immutable once created.
        """
    return _mupdf.FzColorspace_fz_new_indexed_colorspace(self, high, lookup)