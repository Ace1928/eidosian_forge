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
def ll_pdf_dict_get_put_drop(dict, key, val):
    """
    Wrapper for out-params of pdf_dict_get_put_drop().
    Returns: ::pdf_obj *old_val
    """
    outparams = ll_pdf_dict_get_put_drop_outparams()
    ret = ll_pdf_dict_get_put_drop_outparams_fn(dict, key, val, outparams)
    return outparams.old_val