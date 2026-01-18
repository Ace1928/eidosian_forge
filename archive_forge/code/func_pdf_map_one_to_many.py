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
def pdf_map_one_to_many(self, one, many, len):
    """
        Class-aware wrapper for `::pdf_map_one_to_many()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_map_one_to_many(unsigned int one, size_t len)` => int many
        """
    return _mupdf.PdfCmap_pdf_map_one_to_many(self, one, many, len)