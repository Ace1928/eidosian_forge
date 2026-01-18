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
def fz_save_accelerator(self, accel):
    """
        Class-aware wrapper for `::fz_save_accelerator()`.
        	Save accelerator data for the document to a given file.
        """
    return _mupdf.FzDocument_fz_save_accelerator(self, accel)