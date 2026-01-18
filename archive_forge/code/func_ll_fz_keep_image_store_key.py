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
def ll_fz_keep_image_store_key(image):
    """
    Low-level wrapper for `::fz_keep_image_store_key()`.
    Increment the store key reference for an image. Returns the same
    pointer. (This is the count of references for an image held by
    keys in the image store).

    Never throws exceptions.
    """
    return _mupdf.ll_fz_keep_image_store_key(image)