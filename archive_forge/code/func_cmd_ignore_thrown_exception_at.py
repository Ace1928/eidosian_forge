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
def cmd_ignore_thrown_exception_at(self, py_db, cmd_id, seq, text):
    if text:
        replace = 'REPLACE:'
        if text.startswith(replace):
            text = text[8:]
            py_db.filename_to_lines_where_exceptions_are_ignored.clear()
        if text:
            for line in text.split('||'):
                original_filename, line_number = line.split('|')
                original_filename = self.api.filename_to_server(original_filename)
                canonical_normalized_filename = pydevd_file_utils.canonical_normalized_path(original_filename)
                absolute_filename = pydevd_file_utils.absolute_path(original_filename)
                if os.path.exists(absolute_filename):
                    lines_ignored = py_db.filename_to_lines_where_exceptions_are_ignored.get(canonical_normalized_filename)
                    if lines_ignored is None:
                        lines_ignored = py_db.filename_to_lines_where_exceptions_are_ignored[canonical_normalized_filename] = {}
                    lines_ignored[int(line_number)] = 1
                else:
                    sys.stderr.write('pydev debugger: warning: trying to ignore exception thrown on file that does not exist: %s (will have no effect)\n' % (absolute_filename,))