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
def ll_fz_count_chapters(doc):
    """
    Low-level wrapper for `::fz_count_chapters()`.
    Return the number of chapters in the document.
    At least 1.
    """
    return _mupdf.ll_fz_count_chapters(doc)