import sys  # @NoMove
import os
from _pydevd_bundle import pydevd_constants
import atexit
import dis
import io
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
import itertools
import traceback
import weakref
import getpass as getpass_mod
import functools
import pydevd_file_utils
from _pydev_bundle import pydev_imports, pydev_log
from _pydev_bundle._pydev_filesystem_encoding import getfilesystemencoding
from _pydev_bundle.pydev_is_thread_alive import is_thread_alive
from _pydev_bundle.pydev_override import overrides
from _pydev_bundle._pydev_saved_modules import threading, time, thread
from _pydevd_bundle import pydevd_extension_utils, pydevd_frame_utils
from _pydevd_bundle.pydevd_filtering import FilesFiltering, glob_matches_path
from _pydevd_bundle import pydevd_io, pydevd_vm_type, pydevd_defaults
from _pydevd_bundle import pydevd_utils
from _pydevd_bundle import pydevd_runpy
from _pydev_bundle.pydev_console_utils import DebugConsoleStdIn
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle.pydevd_breakpoints import ExceptionBreakpoint, get_exception_breakpoint
from _pydevd_bundle.pydevd_comm_constants import (CMD_THREAD_SUSPEND, CMD_STEP_INTO, CMD_SET_BREAK,
from _pydevd_bundle.pydevd_constants import (get_thread_id, get_current_thread_id,
from _pydevd_bundle.pydevd_defaults import PydevdCustomization  # Note: import alias used on pydev_monkey.
from _pydevd_bundle.pydevd_custom_frames import CustomFramesContainer, custom_frames_container_init
from _pydevd_bundle.pydevd_dont_trace_files import DONT_TRACE, PYDEV_FILE, LIB_FILE, DONT_TRACE_DIRS
from _pydevd_bundle.pydevd_extension_api import DebuggerEventHandler
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame, remove_exception_from_frame
from _pydevd_bundle.pydevd_net_command_factory_xml import NetCommandFactory
from _pydevd_bundle.pydevd_trace_dispatch import (
from _pydevd_bundle.pydevd_utils import save_main_module, is_current_thread_main_thread, \
from _pydevd_frame_eval.pydevd_frame_eval_main import (
import pydev_ipython  # @UnusedImport
from _pydevd_bundle.pydevd_source_mapping import SourceMapping
from _pydevd_bundle.pydevd_concurrency_analyser.pydevd_concurrency_logger import ThreadingLogger, AsyncioLogger, send_concurrency_message, cur_time
from _pydevd_bundle.pydevd_concurrency_analyser.pydevd_thread_wrappers import wrap_threads
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame, NORM_PATHS_AND_BASE_CONTAINER
from pydevd_file_utils import get_fullname, get_package_dir
from os.path import abspath as os_path_abspath
import pydevd_tracing
from _pydevd_bundle.pydevd_comm import (InternalThreadCommand, InternalThreadCommandForAnyThread,
from _pydevd_bundle.pydevd_comm import(InternalConsoleExec,
from _pydevd_bundle.pydevd_daemon_thread import PyDBDaemonThread, mark_as_pydevd_daemon_thread
from _pydevd_bundle.pydevd_process_net_command_json import PyDevJsonCommandProcessor
from _pydevd_bundle.pydevd_process_net_command import process_net_command
from _pydevd_bundle.pydevd_net_command import NetCommand, NULL_NET_COMMAND
from _pydevd_bundle.pydevd_breakpoints import stop_on_unhandled_exception
from _pydevd_bundle.pydevd_collect_bytecode_info import collect_try_except_info, collect_return_info, collect_try_except_info_from_source
from _pydevd_bundle.pydevd_suspended_frames import SuspendedFramesManager
from socket import SHUT_RDWR
from _pydevd_bundle.pydevd_api import PyDevdAPI
from _pydevd_bundle.pydevd_timeout import TimeoutTracker
from _pydevd_bundle.pydevd_thread_lifecycle import suspend_all_threads, mark_thread_suspended
from _pydevd_bundle.pydevd_plugin_utils import PluginManager
def apply_files_filter(self, frame, original_filename, force_check_project_scope):
    """
        Should only be called if `self.is_files_filter_enabled == True` or `force_check_project_scope == True`.

        Note that it covers both the filter by specific paths includes/excludes as well
        as the check which filters out libraries if not in the project scope.

        :param original_filename:
            Note can either be the original filename or the absolute version of that filename.

        :param force_check_project_scope:
            Check that the file is in the project scope even if the global setting
            is off.

        :return bool:
            True if it should be excluded when stepping and False if it should be
            included.
        """
    cache_key = (frame.f_code.co_firstlineno, original_filename, force_check_project_scope, frame.f_code)
    try:
        return self._apply_filter_cache[cache_key]
    except KeyError:
        if self.plugin is not None and (self.has_plugin_line_breaks or self.has_plugin_exception_breaks):
            if not self.plugin.can_skip(self, frame):
                pydev_log.debug_once('File traced (included by plugins): %s', original_filename)
                self._apply_filter_cache[cache_key] = False
                return False
        if self._exclude_filters_enabled:
            absolute_filename = pydevd_file_utils.absolute_path(original_filename)
            exclude_by_filter = self._exclude_by_filter(frame, absolute_filename)
            if exclude_by_filter is not None:
                if exclude_by_filter:
                    pydev_log.debug_once('File not traced (excluded by filters): %s', original_filename)
                    self._apply_filter_cache[cache_key] = True
                    return True
                else:
                    pydev_log.debug_once('File traced (explicitly included by filters): %s', original_filename)
                    self._apply_filter_cache[cache_key] = False
                    return False
        if (self._is_libraries_filter_enabled or force_check_project_scope) and (not self.in_project_scope(frame)):
            self._apply_filter_cache[cache_key] = True
            if force_check_project_scope:
                pydev_log.debug_once('File not traced (not in project): %s', original_filename)
            else:
                pydev_log.debug_once('File not traced (not in project - force_check_project_scope): %s', original_filename)
            return True
        if force_check_project_scope:
            pydev_log.debug_once('File traced: %s (force_check_project_scope)', original_filename)
        else:
            pydev_log.debug_once('File traced: %s', original_filename)
        self._apply_filter_cache[cache_key] = False
        return False