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
def on_stepintargets_request(self, py_db, request):
    """
        :param StepInTargetsRequest request:
        """
    frame_id = request.arguments.frameId
    thread_id = py_db.suspended_frames_manager.get_thread_id_for_variable_reference(frame_id)
    if thread_id is None:
        body = StepInTargetsResponseBody([])
        variables_response = pydevd_base_schema.build_response(request, kwargs={'body': body, 'success': False, 'message': 'Unable to get thread_id from frame_id (thread to get step in targets seems to have resumed already).'})
        return NetCommand(CMD_RETURN, 0, variables_response, is_json=True)
    py_db.post_method_as_internal_command(thread_id, internal_get_step_in_targets_json, request.seq, thread_id, frame_id, request, set_additional_thread_info)