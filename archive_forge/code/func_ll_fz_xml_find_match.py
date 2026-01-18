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
def ll_fz_xml_find_match(item, tag, att, match):
    """
    Low-level wrapper for `::fz_xml_find_match()`.
    Search the siblings of XML nodes starting with item looking for
    the first with the given tag (or any tag if tag is NULL), and
    with a matching attribute.

    Return NULL if none found.
    """
    return _mupdf.ll_fz_xml_find_match(item, tag, att, match)