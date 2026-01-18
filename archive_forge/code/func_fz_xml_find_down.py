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
def fz_xml_find_down(self, tag):
    """
        Class-aware wrapper for `::fz_xml_find_down()`.
        	Search the siblings of XML nodes starting with the first child
        	of item looking for the first with the given tag.

        	Return NULL if none found.
        """
    return _mupdf.FzXml_fz_xml_find_down(self, tag)