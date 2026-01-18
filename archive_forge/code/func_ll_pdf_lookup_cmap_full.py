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
def ll_pdf_lookup_cmap_full(cmap, cpt):
    """
    Wrapper for out-params of pdf_lookup_cmap_full().
    Returns: int, int out
    """
    outparams = ll_pdf_lookup_cmap_full_outparams()
    ret = ll_pdf_lookup_cmap_full_outparams_fn(cmap, cpt, outparams)
    return (ret, outparams.out)