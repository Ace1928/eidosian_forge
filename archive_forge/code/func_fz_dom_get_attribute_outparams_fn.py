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
def fz_dom_get_attribute_outparams_fn(elt, i):
    """
    Class-aware helper for out-params of fz_dom_get_attribute() [fz_dom_get_attribute()].
    """
    ret, att = ll_fz_dom_get_attribute(elt.m_internal, i)
    return (ret, att)