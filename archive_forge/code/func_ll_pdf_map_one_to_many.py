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
def ll_pdf_map_one_to_many(cmap, one, len):
    """
    Wrapper for out-params of pdf_map_one_to_many().
    Returns: int many
    """
    outparams = ll_pdf_map_one_to_many_outparams()
    ret = ll_pdf_map_one_to_many_outparams_fn(cmap, one, len, outparams)
    return outparams.many