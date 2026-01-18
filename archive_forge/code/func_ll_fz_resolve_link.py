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
def ll_fz_resolve_link(doc, uri):
    """
    Wrapper for out-params of fz_resolve_link().
    Returns: fz_location, float xp, float yp
    """
    outparams = ll_fz_resolve_link_outparams()
    ret = ll_fz_resolve_link_outparams_fn(doc, uri, outparams)
    return (ret, outparams.xp, outparams.yp)