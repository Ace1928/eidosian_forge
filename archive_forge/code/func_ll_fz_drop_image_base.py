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
def ll_fz_drop_image_base(image):
    """
    Low-level wrapper for `::fz_drop_image_base()`.
    Internal destructor for the base image class members.

    Exposed to allow derived image classes to be written.
    """
    return _mupdf.ll_fz_drop_image_base(image)