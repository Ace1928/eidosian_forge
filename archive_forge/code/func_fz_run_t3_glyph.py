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
def fz_run_t3_glyph(self, gid, trm, dev):
    """
        Class-aware wrapper for `::fz_run_t3_glyph()`.
        	Run a glyph from a Type3 font to
        	a given device.

        	font: The font to find the glyph in.

        	gid: The glyph to run.

        	trm: The transform to apply.

        	dev: The device to render onto.
        """
    return _mupdf.FzFont_fz_run_t3_glyph(self, gid, trm, dev)