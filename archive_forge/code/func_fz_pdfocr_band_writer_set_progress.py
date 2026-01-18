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
def fz_pdfocr_band_writer_set_progress(self, progress_fn, progress_arg):
    """
        Class-aware wrapper for `::fz_pdfocr_band_writer_set_progress()`.
        	Set the progress callback for a pdfocr bandwriter.
        """
    return _mupdf.FzBandWriter_fz_pdfocr_band_writer_set_progress(self, progress_fn, progress_arg)