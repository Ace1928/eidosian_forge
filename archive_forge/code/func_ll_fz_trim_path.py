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
def ll_fz_trim_path(path):
    """
    Low-level wrapper for `::fz_trim_path()`.
    Minimise the internal storage used by a path.

    As paths are constructed, the internal buffers
    grow. To avoid repeated reallocations they
    grow with some spare space. Once a path has
    been fully constructed, this call allows the
    excess space to be trimmed.
    """
    return _mupdf.ll_fz_trim_path(path)