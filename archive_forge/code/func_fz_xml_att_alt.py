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
def fz_xml_att_alt(self, one, two):
    """
        Class-aware wrapper for `::fz_xml_att_alt()`.
        	Return the value of an attribute of an XML node.
        	If the first attribute doesn't exist, try the second.
        	NULL if neither attribute exists.
        """
    return _mupdf.FzXml_fz_xml_att_alt(self, one, two)