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
def ll_fz_lookup_image_type(type):
    """
    Low-level wrapper for `::fz_lookup_image_type()`.
    Map from (case sensitive) image type string to FZ_IMAGE_*
    type value.
    """
    return _mupdf.ll_fz_lookup_image_type(type)