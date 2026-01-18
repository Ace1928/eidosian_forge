import linecache
import os
from _pydev_bundle.pydev_imports import _queue
from _pydev_bundle._pydev_saved_modules import time
from _pydev_bundle._pydev_saved_modules import threading
from _pydev_bundle._pydev_saved_modules import socket as socket_module
from _pydevd_bundle.pydevd_constants import (DebugInfoHolder, IS_WINDOWS, IS_JYTHON, IS_WASM,
from _pydev_bundle.pydev_override import overrides
import weakref
from _pydev_bundle._pydev_completer import extract_token_and_qualifier
from _pydevd_bundle._debug_adapter.pydevd_schema import VariablesResponseBody, \
from _pydevd_bundle._debug_adapter import pydevd_base_schema, pydevd_schema
from _pydevd_bundle.pydevd_net_command import NetCommand
from _pydevd_bundle.pydevd_xml import ExceptionOnEvaluate
from _pydevd_bundle.pydevd_constants import ForkSafeLock, NULL
from _pydevd_bundle.pydevd_daemon_thread import PyDBDaemonThread
from _pydevd_bundle.pydevd_thread_lifecycle import pydevd_find_thread_by_id, resume_threads
from _pydevd_bundle.pydevd_dont_trace_files import PYDEV_FILE
import dis
import pydevd_file_utils
import itertools
from urllib.parse import quote_plus, unquote_plus
import pydevconsole
from _pydevd_bundle import pydevd_vars, pydevd_io, pydevd_reload
from _pydevd_bundle import pydevd_bytecode_utils
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle import pydevd_vm_type
import sys
import traceback
from _pydevd_bundle.pydevd_utils import quote_smart as quote, compare_object_attrs_key, \
from _pydev_bundle import pydev_log, fsnotify
from _pydev_bundle.pydev_log import exception as pydev_log_exception
from _pydev_bundle import _pydev_completer
from pydevd_tracing import get_exception_traceback_str
from _pydevd_bundle import pydevd_console
from _pydev_bundle.pydev_monkey import disable_trace_thread_modules, enable_trace_thread_modules
from io import StringIO
from _pydevd_bundle.pydevd_comm_constants import *  # @UnusedWildImport
def internal_change_variable_json(py_db, request):
    """
    The pydevd_vars.change_attr_expression(thread_id, frame_id, attr, value, dbg) can only
    deal with changing at a frame level, so, currently changing the contents of something
    in a different scope is currently not supported.

    :param SetVariableRequest request:
    """
    arguments = request.arguments
    variables_reference = arguments.variablesReference
    scope = None
    if isinstance_checked(variables_reference, ScopeRequest):
        scope = variables_reference
        variables_reference = variables_reference.variable_reference
    fmt = arguments.format
    if hasattr(fmt, 'to_dict'):
        fmt = fmt.to_dict()
    try:
        variable = py_db.suspended_frames_manager.get_variable(variables_reference)
    except KeyError:
        variable = None
    if variable is None:
        _write_variable_response(py_db, request, value='', success=False, message='Unable to find variable container to change: %s.' % (variables_reference,))
        return
    child_var = variable.change_variable(arguments.name, arguments.value, py_db, fmt=fmt)
    if child_var is None:
        _write_variable_response(py_db, request, value='', success=False, message='Unable to change: %s.' % (arguments.name,))
        return
    var_data = child_var.get_var_data(fmt=fmt)
    body = SetVariableResponseBody(value=var_data['value'], type=var_data['type'], variablesReference=var_data.get('variablesReference'), namedVariables=var_data.get('namedVariables'), indexedVariables=var_data.get('indexedVariables'))
    variables_response = pydevd_base_schema.build_response(request, kwargs={'body': body})
    py_db.writer.add_command(NetCommand(CMD_RETURN, 0, variables_response, is_json=True))