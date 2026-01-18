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
def fz_try_parse_xml_archive_entry(self, filename, preserve_white):
    """
        Class-aware wrapper for `::fz_try_parse_xml_archive_entry()`.
        	Try and parse the contents of an archive entry into a tree of xml nodes.

        	preserve_white: whether to keep or delete all-whitespace nodes.

        	Will return NULL if the archive entry can't be found. Otherwise behaves
        	the same as fz_parse_xml_archive_entry. May throw exceptions.
        """
    return _mupdf.FzArchive_fz_try_parse_xml_archive_entry(self, filename, preserve_white)