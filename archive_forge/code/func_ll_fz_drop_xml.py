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
def ll_fz_drop_xml(xml):
    """
    Low-level wrapper for `::fz_drop_xml()`.
    Drop a reference to the XML. When the last reference is
    dropped, the node and all its children and siblings will
    be freed.
    """
    return _mupdf.ll_fz_drop_xml(xml)