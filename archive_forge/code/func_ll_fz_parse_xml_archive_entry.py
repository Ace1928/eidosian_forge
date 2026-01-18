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
def ll_fz_parse_xml_archive_entry(dir, filename, preserve_white):
    """
    Low-level wrapper for `::fz_parse_xml_archive_entry()`.
    Parse the contents of an archive entry into a tree of xml nodes.

    preserve_white: whether to keep or delete all-whitespace nodes.
    """
    return _mupdf.ll_fz_parse_xml_archive_entry(dir, filename, preserve_white)