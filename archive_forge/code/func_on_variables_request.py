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
def on_variables_request(self, py_db, request):
    """
        Variables can be asked whenever some place returned a variables reference (so, it
        can be a scope gotten from on_scopes_request, the result of some evaluation, etc.).

        Note that in the DAP the variables reference requires a unique int... the way this works for
        pydevd is that an instance is generated for that specific variable reference and we use its
        id(instance) to identify it to make sure all items are unique (and the actual {id->instance}
        is added to a dict which is only valid while the thread is suspended and later cleared when
        the related thread resumes execution).

        see: SuspendedFramesManager

        :param VariablesRequest request:
        """
    arguments = request.arguments
    variables_reference = arguments.variablesReference
    if isinstance(variables_reference, ScopeRequest):
        variables_reference = variables_reference.variable_reference
    thread_id = py_db.suspended_frames_manager.get_thread_id_for_variable_reference(variables_reference)
    if thread_id is not None:
        self.api.request_get_variable_json(py_db, request, thread_id)
    else:
        variables = []
        body = VariablesResponseBody(variables)
        variables_response = pydevd_base_schema.build_response(request, kwargs={'body': body, 'success': False, 'message': 'Unable to find thread to evaluate variable reference.'})
        return NetCommand(CMD_RETURN, 0, variables_response, is_json=True)