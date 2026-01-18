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
def ll_fz_colorspace_device_n_has_cmyk(cs):
    """
    Low-level wrapper for `::fz_colorspace_device_n_has_cmyk()`.
    True if DeviceN color space has cyan magenta yellow or black as
    one of its colorants.
    """
    return _mupdf.ll_fz_colorspace_device_n_has_cmyk(cs)