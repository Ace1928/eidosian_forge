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
def ll_fz_xml_text(item):
    """
    Low-level wrapper for `::fz_xml_text()`.
    Return the text content of an XML node.
    Return NULL if the node is a tag.
    """
    return _mupdf.ll_fz_xml_text(item)