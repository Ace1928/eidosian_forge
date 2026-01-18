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
def ll_fz_pdfocr_band_writer_set_progress(writer, progress_fn, progress_arg):
    """
    Low-level wrapper for `::fz_pdfocr_band_writer_set_progress()`.
    Set the progress callback for a pdfocr bandwriter.
    """
    return _mupdf.ll_fz_pdfocr_band_writer_set_progress(writer, progress_fn, progress_arg)