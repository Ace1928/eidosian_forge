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
def ll_fz_write_image_as_data_uri(out, image):
    """
    Low-level wrapper for `::fz_write_image_as_data_uri()`.
    Write image as a data URI (for HTML and SVG output).
    """
    return _mupdf.ll_fz_write_image_as_data_uri(out, image)