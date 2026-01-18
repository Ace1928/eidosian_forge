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
def fz_measure_string(self, trm, s, wmode, bidi_level, markup_dir, language):
    """
        Class-aware wrapper for `::fz_measure_string()`.
        	Measure the advance width of a UTF8 string should it be added to a text object.

        	This uses the same layout algorithms as fz_show_string, and can be used
        	to calculate text alignment adjustments.
        """
    return _mupdf.FzFont_fz_measure_string(self, trm, s, wmode, bidi_level, markup_dir, language)