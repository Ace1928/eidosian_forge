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
def fz_new_display_list(self):
    """
        Class-aware wrapper for `::fz_new_display_list()`.
        	Create an empty display list.

        	A display list contains drawing commands (text, images, etc.).
        	Use fz_new_list_device for populating the list.

        	mediabox: Bounds of the page (in points) represented by the
        	display list.
        """
    return _mupdf.FzRect_fz_new_display_list(self)