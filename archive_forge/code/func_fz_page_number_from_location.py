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
def fz_page_number_from_location(self, loc):
    """
        Class-aware wrapper for `::fz_page_number_from_location()`.
        	Converts from chapter+page to page number. This may cause many
        	chapters to be laid out in order to calculate the number of
        	pages within those chapters.
        """
    return _mupdf.FzDocument_fz_page_number_from_location(self, loc)