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
def ll_pdf_js_event_result_validate(js):
    """
    Wrapper for out-params of pdf_js_event_result_validate().
    Returns: int, char *newvalue
    """
    outparams = ll_pdf_js_event_result_validate_outparams()
    ret = ll_pdf_js_event_result_validate_outparams_fn(js, outparams)
    return (ret, outparams.newvalue)