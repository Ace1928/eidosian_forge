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
def fz_dom_get_attribute(self, i, att):
    """
        Class-aware wrapper for `::fz_dom_get_attribute()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_dom_get_attribute(int i)` => `(const char *, const char *att)`

        	Enumerate through the attributes of an element.

        	Call with i=0,1,2,3... to enumerate attributes.

        	On return *att and the return value will be NULL if there are not
        	that many attributes to read. Otherwise, *att will be filled in
        	with a borrowed pointer to the attribute name, and the return
        	value will be a borrowed pointer to the value.
        """
    return _mupdf.FzXml_fz_dom_get_attribute(self, i, att)