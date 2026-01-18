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
def on_stepin_request(self, py_db, request):
    """
        :param StepInRequest request:
        """
    arguments = request.arguments
    thread_id = arguments.threadId
    target_id = arguments.targetId
    if target_id is not None:
        thread = pydevd_find_thread_by_id(thread_id)
        if thread is None:
            response = Response(request_seq=request.seq, success=False, command=request.command, message='Unable to find thread from thread_id: %s' % (thread_id,), body={})
            return NetCommand(CMD_RETURN, 0, response, is_json=True)
        info = set_additional_thread_info(thread)
        target_id_to_smart_step_into_variant = info.target_id_to_smart_step_into_variant
        if not target_id_to_smart_step_into_variant:
            variables_response = pydevd_base_schema.build_response(request, kwargs={'success': False, 'message': 'Unable to step into target (no targets are saved in the thread info).'})
            return NetCommand(CMD_RETURN, 0, variables_response, is_json=True)
        variant = target_id_to_smart_step_into_variant.get(target_id)
        if variant is not None:
            parent = variant.parent
            if parent is not None:
                self.api.request_smart_step_into(py_db, request.seq, thread_id, parent.offset, variant.offset)
            else:
                self.api.request_smart_step_into(py_db, request.seq, thread_id, variant.offset, -1)
        else:
            variables_response = pydevd_base_schema.build_response(request, kwargs={'success': False, 'message': 'Unable to find step into target %s. Available targets: %s' % (target_id, target_id_to_smart_step_into_variant)})
            return NetCommand(CMD_RETURN, 0, variables_response, is_json=True)
    else:
        if py_db.get_use_libraries_filter():
            step_cmd_id = CMD_STEP_INTO_MY_CODE
        else:
            step_cmd_id = CMD_STEP_INTO
        self.api.request_step(py_db, thread_id, step_cmd_id)
    response = pydevd_base_schema.build_response(request)
    return NetCommand(CMD_RETURN, 0, response, is_json=True)