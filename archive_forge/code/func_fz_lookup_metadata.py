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
def fz_lookup_metadata(self, key, buf, size):
    """
        Class-aware wrapper for `::fz_lookup_metadata()`.
        	Retrieve document meta data strings.

        	doc: The document to query.

        	key: Which meta data key to retrieve...

        	Basic information:
        		'format'	-- Document format and version.
        		'encryption'	-- Description of the encryption used.

        	From the document information dictionary:
        		'info:Title'
        		'info:Author'
        		'info:Subject'
        		'info:Keywords'
        		'info:Creator'
        		'info:Producer'
        		'info:CreationDate'
        		'info:ModDate'

        	buf: The buffer to hold the results (a nul-terminated UTF-8
        	string).

        	size: Size of 'buf'.

        	Returns the number of bytes need to store the string plus terminator
        	(will be larger than 'size' if the output was truncated), or -1 if the
        	key is not recognized or found.
        """
    return _mupdf.FzDocument_fz_lookup_metadata(self, key, buf, size)