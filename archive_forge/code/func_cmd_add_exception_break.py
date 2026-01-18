import json
import os
import sys
import traceback
from _pydev_bundle import pydev_log
from _pydev_bundle.pydev_log import exception as pydev_log_exception
from _pydevd_bundle import pydevd_traceproperty, pydevd_dont_trace, pydevd_utils
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle.pydevd_breakpoints import get_exception_class
from _pydevd_bundle.pydevd_comm import (
from _pydevd_bundle.pydevd_constants import NEXT_VALUE_SEPARATOR, IS_WINDOWS, NULL
from _pydevd_bundle.pydevd_comm_constants import ID_TO_MEANING, CMD_EXEC_EXPRESSION, CMD_AUTHENTICATE
from _pydevd_bundle.pydevd_api import PyDevdAPI
from io import StringIO
from _pydevd_bundle.pydevd_net_command import NetCommand
from _pydevd_bundle.pydevd_thread_lifecycle import pydevd_find_thread_by_id
import pydevd_file_utils
def cmd_add_exception_break(self, py_db, cmd_id, seq, text):
    condition = ''
    expression = ''
    if text.find('\t') != -1:
        try:
            exception, condition, expression, notify_on_handled_exceptions, notify_on_unhandled_exceptions, ignore_libraries = text.split('\t', 5)
        except:
            exception, notify_on_handled_exceptions, notify_on_unhandled_exceptions, ignore_libraries = text.split('\t', 3)
    else:
        exception, notify_on_handled_exceptions, notify_on_unhandled_exceptions, ignore_libraries = (text, 0, 0, 0)
    condition = condition.replace('@_@NEW_LINE_CHAR@_@', '\n').replace('@_@TAB_CHAR@_@', '\t').strip()
    if condition is not None and (len(condition) == 0 or condition == 'None'):
        condition = None
    expression = expression.replace('@_@NEW_LINE_CHAR@_@', '\n').replace('@_@TAB_CHAR@_@', '\t').strip()
    if expression is not None and (len(expression) == 0 or expression == 'None'):
        expression = None
    if exception.find('-') != -1:
        breakpoint_type, exception = exception.split('-')
    else:
        breakpoint_type = 'python'
    if breakpoint_type == 'python':
        self.api.add_python_exception_breakpoint(py_db, exception, condition, expression, notify_on_handled_exceptions=int(notify_on_handled_exceptions) > 0, notify_on_unhandled_exceptions=int(notify_on_unhandled_exceptions) == 1, notify_on_user_unhandled_exceptions=0, notify_on_first_raise_only=int(notify_on_handled_exceptions) == 2, ignore_libraries=int(ignore_libraries) > 0)
    else:
        self.api.add_plugins_exception_breakpoint(py_db, breakpoint_type, exception)