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
def ll_pdf_edit_text_field_value(widget, value, change):
    """
    Wrapper for out-params of pdf_edit_text_field_value().
    Returns: int, int selStart, int selEnd, char *newvalue
    """
    outparams = ll_pdf_edit_text_field_value_outparams()
    ret = ll_pdf_edit_text_field_value_outparams_fn(widget, value, change, outparams)
    return (ret, outparams.selStart, outparams.selEnd, outparams.newvalue)