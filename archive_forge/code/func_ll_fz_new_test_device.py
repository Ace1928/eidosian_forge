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
def ll_fz_new_test_device(threshold, options, passthrough):
    """
    Wrapper for out-params of fz_new_test_device().
    Returns: fz_device *, int is_color
    """
    outparams = ll_fz_new_test_device_outparams()
    ret = ll_fz_new_test_device_outparams_fn(threshold, options, passthrough, outparams)
    return (ret, outparams.is_color)