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
def ll_pdf_annot_line_ending_styles(annot):
    """
    Wrapper for out-params of pdf_annot_line_ending_styles().
    Returns: enum pdf_line_ending start_style, enum pdf_line_ending end_style
    """
    outparams = ll_pdf_annot_line_ending_styles_outparams()
    ret = ll_pdf_annot_line_ending_styles_outparams_fn(annot, outparams)
    return (outparams.start_style, outparams.end_style)