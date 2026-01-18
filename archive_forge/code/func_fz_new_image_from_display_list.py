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
def fz_new_image_from_display_list(w, h, list):
    """
    Class-aware wrapper for `::fz_new_image_from_display_list()`.
    	Create a new image from a display list.

    	w, h: The conceptual width/height of the image.

    	transform: The matrix that needs to be applied to the given
    	list to make it render to the unit square.

    	list: The display list.
    """
    return _mupdf.fz_new_image_from_display_list(w, h, list)