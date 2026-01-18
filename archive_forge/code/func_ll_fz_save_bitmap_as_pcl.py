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
def ll_fz_save_bitmap_as_pcl(bitmap, filename, append, pcl):
    """
    Low-level wrapper for `::fz_save_bitmap_as_pcl()`.
    Save a bitmap as mono PCL.
    """
    return _mupdf.ll_fz_save_bitmap_as_pcl(bitmap, filename, append, pcl)