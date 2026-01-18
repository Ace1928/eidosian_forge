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
def fz_store_scavenge_external_outparams_fn(size):
    """
    Class-aware helper for out-params of fz_store_scavenge_external() [fz_store_scavenge_external()].
    """
    ret, phase = ll_fz_store_scavenge_external(size)
    return (ret, phase)