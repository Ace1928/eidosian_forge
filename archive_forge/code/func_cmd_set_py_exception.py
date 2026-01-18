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
def cmd_set_py_exception(self, py_db, cmd_id, seq, text):
    try:
        splitted = text.split(';')
        py_db.break_on_uncaught_exceptions = {}
        py_db.break_on_caught_exceptions = {}
        py_db.break_on_user_uncaught_exceptions = {}
        if len(splitted) >= 5:
            if splitted[0] == 'true':
                break_on_uncaught = True
            else:
                break_on_uncaught = False
            if splitted[1] == 'true':
                break_on_caught = True
            else:
                break_on_caught = False
            if splitted[2] == 'true':
                py_db.skip_on_exceptions_thrown_in_same_context = True
            else:
                py_db.skip_on_exceptions_thrown_in_same_context = False
            if splitted[3] == 'true':
                py_db.ignore_exceptions_thrown_in_lines_with_ignore_exception = True
            else:
                py_db.ignore_exceptions_thrown_in_lines_with_ignore_exception = False
            if splitted[4] == 'true':
                ignore_libraries = True
            else:
                ignore_libraries = False
            for exception_type in splitted[5:]:
                exception_type = exception_type.strip()
                if not exception_type:
                    continue
                py_db.add_break_on_exception(exception_type, condition=None, expression=None, notify_on_handled_exceptions=break_on_caught, notify_on_unhandled_exceptions=break_on_uncaught, notify_on_user_unhandled_exceptions=False, notify_on_first_raise_only=True, ignore_libraries=ignore_libraries)
        else:
            pydev_log.exception("Expected to have at least 5 ';' separated items. Received: %s", text)
    except:
        pydev_log.exception('Error when setting exception list. Received: %s', text)