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
def fz_new_display_list_from_svg(self, base_uri, dir, w, h):
    """
        Class-aware wrapper for `::fz_new_display_list_from_svg()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_new_display_list_from_svg(const char *base_uri, ::fz_archive *dir)` => `(fz_display_list *, float w, float h)`

        	Parse an SVG document into a display-list.
        """
    return _mupdf.FzBuffer_fz_new_display_list_from_svg(self, base_uri, dir, w, h)