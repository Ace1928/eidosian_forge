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
def ll_pdf_decode_cmap(cmap, s, e):
    """
    Wrapper for out-params of pdf_decode_cmap().
    Returns: int, unsigned int cpt
    """
    outparams = ll_pdf_decode_cmap_outparams()
    ret = ll_pdf_decode_cmap_outparams_fn(cmap, s, e, outparams)
    return (ret, outparams.cpt)