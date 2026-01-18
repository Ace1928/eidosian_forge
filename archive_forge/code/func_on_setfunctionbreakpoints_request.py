import itertools
import json
import linecache
import os
import platform
import sys
from functools import partial
import pydevd_file_utils
from _pydev_bundle import pydev_log
from _pydevd_bundle._debug_adapter import pydevd_base_schema, pydevd_schema
from _pydevd_bundle._debug_adapter.pydevd_schema import (
from _pydevd_bundle.pydevd_api import PyDevdAPI
from _pydevd_bundle.pydevd_breakpoints import get_exception_class, FunctionBreakpoint
from _pydevd_bundle.pydevd_comm_constants import (
from _pydevd_bundle.pydevd_filtering import ExcludeFilter
from _pydevd_bundle.pydevd_json_debug_options import _extract_debug_options, DebugOptions
from _pydevd_bundle.pydevd_net_command import NetCommand
from _pydevd_bundle.pydevd_utils import convert_dap_log_message_to_expression, ScopeRequest
from _pydevd_bundle.pydevd_constants import (PY_IMPL_NAME, DebugInfoHolder, PY_VERSION_STR,
from _pydevd_bundle.pydevd_trace_dispatch import USING_CYTHON
from _pydevd_frame_eval.pydevd_frame_eval_main import USING_FRAME_EVAL
from _pydevd_bundle.pydevd_comm import internal_get_step_in_targets_json
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle.pydevd_thread_lifecycle import pydevd_find_thread_by_id
def on_setfunctionbreakpoints_request(self, py_db, request):
    """
        :param SetFunctionBreakpointsRequest request:
        """
    response = self._verify_launch_or_attach_done(request)
    if response is not None:
        return response
    arguments = request.arguments
    function_breakpoints = []
    suspend_policy = 'ALL'
    is_logpoint = False
    expression = None
    breakpoints_set = []
    for bp in arguments.breakpoints:
        hit_condition = self._get_hit_condition_expression(bp.get('hitCondition'))
        condition = bp.get('condition')
        function_breakpoints.append(FunctionBreakpoint(bp['name'], condition, expression, suspend_policy, hit_condition, is_logpoint))
        breakpoints_set.append(pydevd_schema.Breakpoint(verified=True, id=self._next_breakpoint_id()).to_dict())
    self.api.set_function_breakpoints(py_db, function_breakpoints)
    body = {'breakpoints': breakpoints_set}
    set_breakpoints_response = pydevd_base_schema.build_response(request, kwargs={'body': body})
    return NetCommand(CMD_RETURN, 0, set_breakpoints_response, is_json=True)