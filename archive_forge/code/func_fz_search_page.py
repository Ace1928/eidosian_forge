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
def fz_search_page(self, needle, hit_mark, hit_bbox, hit_max):
    """
        Class-aware wrapper for `::fz_search_page()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_search_page(const char *needle, ::fz_quad *hit_bbox, int hit_max)` => `(int, int hit_mark)`

        	Search for the 'needle' text on the page.
        	Record the hits in the hit_bbox array and return the number of
        	hits. Will stop looking once it has filled hit_max rectangles.
        """
    return _mupdf.FzPage_fz_search_page(self, needle, hit_mark, hit_bbox, hit_max)