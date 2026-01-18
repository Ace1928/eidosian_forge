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
def fz_clamp_color(self, _in, out):
    """
        Class-aware wrapper for `::fz_clamp_color()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_clamp_color(const float *in)` => float out

        	Clamp the samples in a color to the correct ranges for a
        	given colorspace.
        """
    return _mupdf.FzColorspace_fz_clamp_color(self, _in, out)