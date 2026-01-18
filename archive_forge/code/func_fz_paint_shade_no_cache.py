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
def fz_paint_shade_no_cache(self, override_cs, ctm, dest, color_params, bbox, eop):
    """ Extra wrapper for fz_paint_shade(), passing cache=NULL."""
    return _mupdf.FzShade_fz_paint_shade_no_cache(self, override_cs, ctm, dest, color_params, bbox, eop)