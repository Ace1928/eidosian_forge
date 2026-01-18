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
def ll_pdf_undoredo_state(doc):
    """
    Wrapper for out-params of pdf_undoredo_state().
    Returns: int, int steps
    """
    outparams = ll_pdf_undoredo_state_outparams()
    ret = ll_pdf_undoredo_state_outparams_fn(doc, outparams)
    return (ret, outparams.steps)