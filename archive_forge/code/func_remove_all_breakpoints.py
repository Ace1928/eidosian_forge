import sys
import bisect
import types
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle import pydevd_utils, pydevd_source_mapping
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle.pydevd_comm import (InternalGetThreadStack, internal_get_completions,
from _pydevd_bundle.pydevd_comm_constants import (CMD_THREAD_SUSPEND, file_system_encoding,
from _pydevd_bundle.pydevd_constants import (get_current_thread_id, set_protocol, get_protocol,
from _pydevd_bundle.pydevd_net_command_factory_json import NetCommandFactoryJson
from _pydevd_bundle.pydevd_net_command_factory_xml import NetCommandFactory
import pydevd_file_utils
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_breakpoints import LineBreakpoint
from pydevd_tracing import get_exception_traceback_str
import os
import subprocess
import ctypes
from _pydevd_bundle.pydevd_collect_bytecode_info import code_to_bytecode_representation
import itertools
import linecache
from _pydevd_bundle.pydevd_utils import DAPGrouper, interrupt_main_thread
from _pydevd_bundle.pydevd_daemon_thread import run_as_pydevd_daemon_thread
from _pydevd_bundle.pydevd_thread_lifecycle import pydevd_find_thread_by_id, resume_threads
import tokenize
def remove_all_breakpoints(self, py_db, received_filename):
    """
        Removes all the breakpoints from a given file or from all files if received_filename == '*'.

        :param str received_filename:
            Note: must be sent as it was received in the protocol. It may be translated in this
            function.
        """
    assert received_filename.__class__ == str
    changed = False
    lst = [py_db.file_to_id_to_line_breakpoint, py_db.file_to_id_to_plugin_breakpoint, py_db.breakpoints]
    if hasattr(py_db, 'django_breakpoints'):
        lst.append(py_db.django_breakpoints)
    if hasattr(py_db, 'jinja2_breakpoints'):
        lst.append(py_db.jinja2_breakpoints)
    if received_filename == '*':
        py_db.api_received_breakpoints.clear()
        for file_to_id_to_breakpoint in lst:
            if file_to_id_to_breakpoint:
                file_to_id_to_breakpoint.clear()
                changed = True
    else:
        received_filename_normalized = pydevd_file_utils.normcase_from_client(received_filename)
        items = list(py_db.api_received_breakpoints.items())
        translated_filenames = []
        for key, val in items:
            original_filename_normalized, _breakpoint_id = key
            if original_filename_normalized == received_filename_normalized:
                canonical_normalized_filename, _api_add_breakpoint_params = val
                translated_filenames.append(canonical_normalized_filename)
                del py_db.api_received_breakpoints[key]
        for canonical_normalized_filename in translated_filenames:
            for file_to_id_to_breakpoint in lst:
                if canonical_normalized_filename in file_to_id_to_breakpoint:
                    file_to_id_to_breakpoint.pop(canonical_normalized_filename, None)
                    changed = True
    if changed:
        py_db.on_breakpoints_changed(removed=True)