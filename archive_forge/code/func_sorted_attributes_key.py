from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_utils import hasattr_checked, DAPGrouper, Timer
from io import StringIO
import traceback
from os.path import basename
from functools import partial
from _pydevd_bundle.pydevd_constants import IS_PY36_OR_GREATER, \
from _pydevd_bundle.pydevd_safe_repr import SafeRepr
from _pydevd_bundle import pydevd_constants
def sorted_attributes_key(attr_name):
    if attr_name.startswith('__'):
        if attr_name.endswith('__'):
            return (3, attr_name)
        else:
            return (2, attr_name)
    elif attr_name.startswith('_'):
        return (1, attr_name)
    else:
        return (0, attr_name)