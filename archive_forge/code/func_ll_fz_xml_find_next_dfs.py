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
def ll_fz_xml_find_next_dfs(item, tag, att, match):
    """
    Low-level wrapper for `::fz_xml_find_next_dfs()`.
    Perform a depth first search onwards from item, returning the first
    child that matches the given tag (or any tag if tag is NULL),
    with the given attribute (if att is non NULL), that matches
    match (if match is non NULL).
    """
    return _mupdf.ll_fz_xml_find_next_dfs(item, tag, att, match)