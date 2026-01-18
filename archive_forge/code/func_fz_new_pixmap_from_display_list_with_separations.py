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
def fz_new_pixmap_from_display_list_with_separations(self, ctm, cs, seps, alpha):
    """
        Class-aware wrapper for `::fz_new_pixmap_from_display_list_with_separations()`.
        	Render the page contents with control over spot colors.

        	Ownership of the pixmap is returned to the caller.
        """
    return _mupdf.FzDisplayList_fz_new_pixmap_from_display_list_with_separations(self, ctm, cs, seps, alpha)