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
def process_net_command_json(self, py_db, json_contents, send_response=True):
    """
        Processes a debug adapter protocol json command.
        """
    DEBUG = False
    try:
        if isinstance(json_contents, bytes):
            json_contents = json_contents.decode('utf-8')
        request = self.from_json(json_contents, update_ids_from_dap=True)
    except Exception as e:
        try:
            loaded_json = json.loads(json_contents)
            request = Request(loaded_json.get('command', '<unknown>'), loaded_json['seq'])
        except:
            pydev_log.exception('Error loading json: %s', json_contents)
            return
        error_msg = str(e)
        if error_msg.startswith("'") and error_msg.endswith("'"):
            error_msg = error_msg[1:-1]

        def on_request(py_db, request):
            error_response = {'type': 'response', 'request_seq': request.seq, 'success': False, 'command': request.command, 'message': error_msg}
            return NetCommand(CMD_RETURN, 0, error_response, is_json=True)
    else:
        if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 1:
            pydev_log.info('Process %s: %s\n' % (request.__class__.__name__, json.dumps(request.to_dict(update_ids_to_dap=True), indent=4, sort_keys=True)))
        assert request.type == 'request'
        method_name = 'on_%s_request' % (request.command.lower(),)
        on_request = getattr(self, method_name, None)
        if on_request is None:
            print('Unhandled: %s not available in PyDevJsonCommandProcessor.\n' % (method_name,))
            return
        if DEBUG:
            print('Handled in pydevd: %s (in PyDevJsonCommandProcessor).\n' % (method_name,))
    with py_db._main_lock:
        if request.__class__ == PydevdAuthorizeRequest:
            authorize_request = request
            access_token = authorize_request.arguments.debugServerAccessToken
            py_db.authentication.login(access_token)
        if not py_db.authentication.is_authenticated():
            response = Response(request.seq, success=False, command=request.command, message='Client not authenticated.', body={})
            cmd = NetCommand(CMD_RETURN, 0, response, is_json=True)
            py_db.writer.add_command(cmd)
            return
        cmd = on_request(py_db, request)
        if cmd is not None and send_response:
            py_db.writer.add_command(cmd)