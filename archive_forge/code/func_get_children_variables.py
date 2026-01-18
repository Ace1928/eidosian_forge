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
@silence_warnings_decorator
@overrides(_AbstractVariable.get_children_variables)
def get_children_variables(self, fmt=None, scope=None):
    children_variables = []
    if scope is not None:
        assert isinstance(scope, ScopeRequest)
        scope = scope.scope
    if scope in ('locals', None):
        dct = self.frame.f_locals
    elif scope == 'globals':
        dct = self.frame.f_globals
    else:
        raise AssertionError('Unexpected scope: %s' % (scope,))
    lst, group_entries = self._group_entries([(x[0], x[1], None) for x in list(dct.items()) if x[0] != '_pydev_stop_at_break'], handle_return_values=True)
    group_variables = []
    for key, val, _ in group_entries:
        val.contents_debug_adapter_protocol.sort(key=lambda v: sorted_attributes_key(v[0]))
        variable = _ObjectVariable(self.py_db, key, val, self._register_variable, False, key, frame=self.frame)
        group_variables.append(variable)
    for key, val, _ in lst:
        is_return_value = key == RETURN_VALUES_DICT
        if is_return_value:
            for return_key, return_value in val.items():
                variable = _ObjectVariable(self.py_db, return_key, return_value, self._register_variable, is_return_value, '%s[%r]' % (key, return_key), frame=self.frame)
                children_variables.append(variable)
        else:
            variable = _ObjectVariable(self.py_db, key, val, self._register_variable, is_return_value, key, frame=self.frame)
            children_variables.append(variable)
    children_variables.sort(key=sorted_variables_key)
    if group_variables:
        children_variables = group_variables + children_variables
    return children_variables