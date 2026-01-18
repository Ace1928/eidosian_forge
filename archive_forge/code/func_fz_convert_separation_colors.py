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
def fz_convert_separation_colors(self, src_color, dst_seps, dst_cs, dst_color, color_params):
    """
        Class-aware wrapper for `::fz_convert_separation_colors()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_convert_separation_colors(const float *src_color, ::fz_separations *dst_seps, ::fz_colorspace *dst_cs, ::fz_color_params color_params)` => float dst_color

        	Convert a color given in terms of one colorspace,
        	to a color in terms of another colorspace/separations.
        """
    return _mupdf.FzColorspace_fz_convert_separation_colors(self, src_color, dst_seps, dst_cs, dst_color, color_params)