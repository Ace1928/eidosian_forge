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
def fz_paint_shade(self, override_cs, ctm, dest, color_params, bbox, eop, cache):
    """
        Class-aware wrapper for `::fz_paint_shade()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_paint_shade(::fz_colorspace *override_cs, ::fz_matrix ctm, ::fz_pixmap *dest, ::fz_color_params color_params, ::fz_irect bbox, const ::fz_overprint *eop, ::fz_shade_color_cache **cache)` =>

        	Render a shade to a given pixmap.

        	shade: The shade to paint.

        	override_cs: NULL, or colorspace to override the shades
        	inbuilt colorspace.

        	ctm: The transform to apply.

        	dest: The pixmap to render into.

        	color_params: The color rendering settings

        	bbox: Pointer to a bounding box to limit the rendering
        	of the shade.

        	eop: NULL, or pointer to overprint bitmap.

        	cache: *cache is used to cache color information. If *cache is NULL it
        	is set to point to a new fz_shade_color_cache. If cache is NULL it is
        	ignored.
        """
    return _mupdf.FzShade_fz_paint_shade(self, override_cs, ctm, dest, color_params, bbox, eop, cache)