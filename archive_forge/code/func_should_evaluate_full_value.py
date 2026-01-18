from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_extension_utils
from _pydevd_bundle import pydevd_resolver
import sys
from _pydevd_bundle.pydevd_constants import BUILTINS_MODULE_NAME, MAXIMUM_VARIABLE_REPRESENTATION_SIZE, \
from _pydev_bundle.pydev_imports import quote
from _pydevd_bundle.pydevd_extension_api import TypeResolveProvider, StrPresentationProvider
from _pydevd_bundle.pydevd_utils import isinstance_checked, hasattr_checked, DAPGrouper
from _pydevd_bundle.pydevd_resolver import get_var_scope, MoreItems, MoreItemsRange
from typing import Optional
def should_evaluate_full_value(val):
    return not LOAD_VALUES_ASYNC or (is_builtin(type(val)) and (not isinstance_checked(val, (list, tuple, dict))))