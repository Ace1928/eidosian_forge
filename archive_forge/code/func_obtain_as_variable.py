from contextlib import contextmanager
import sys
from _pydevd_bundle.pydevd_constants import get_frame, RETURN_VALUES_DICT, \
from _pydevd_bundle.pydevd_xml import get_variable_details, get_type
from _pydev_bundle.pydev_override import overrides
from _pydevd_bundle.pydevd_resolver import sorted_attributes_key, TOO_LARGE_ATTR, get_var_scope
from _pydevd_bundle.pydevd_safe_repr import SafeRepr
from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_vars
from _pydev_bundle.pydev_imports import Exec
from _pydevd_bundle.pydevd_frame_utils import FramesList
from _pydevd_bundle.pydevd_utils import ScopeRequest, DAPGrouper, Timer
from typing import Optional
def obtain_as_variable(self, name, value, evaluate_name=None, frame=None):
    if evaluate_name is None:
        evaluate_name = name
    variable_reference = id(value)
    variable = self._variable_reference_to_variable.get(variable_reference)
    if variable is not None:
        return variable
    return _ObjectVariable(self.py_db, name, value, self._register_variable, is_return_value=False, evaluate_name=evaluate_name, frame=frame)