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
def internal_reload_code(dbg, seq, module_name, filename):
    try:
        found_module_to_reload = False
        if module_name is not None:
            module_name = module_name
            if module_name not in sys.modules:
                if '.' in module_name:
                    new_module_name = module_name.split('.')[-1]
                    if new_module_name in sys.modules:
                        module_name = new_module_name
        modules_to_reload = {}
        module = sys.modules.get(module_name)
        if module is not None:
            modules_to_reload[id(module)] = (module, module_name)
        if filename:
            filename = pydevd_file_utils.normcase(filename)
            for module_name, module in sys.modules.copy().items():
                f = getattr_checked(module, '__file__')
                if f is not None:
                    if f.endswith(('.pyc', '.pyo')):
                        f = f[:-1]
                    if pydevd_file_utils.normcase(f) == filename:
                        modules_to_reload[id(module)] = (module, module_name)
        if not modules_to_reload:
            if filename and module_name:
                _send_io_message(dbg, 'code reload: Unable to find module %s to reload for path: %s\n' % (module_name, filename))
            elif filename:
                _send_io_message(dbg, 'code reload: Unable to find module to reload for path: %s\n' % (filename,))
            elif module_name:
                _send_io_message(dbg, 'code reload: Unable to find module to reload: %s\n' % (module_name,))
        else:
            for module, module_name in modules_to_reload.values():
                _send_io_message(dbg, 'code reload: Start reloading module: "' + module_name + '" ... \n')
                found_module_to_reload = True
                if pydevd_reload.xreload(module):
                    _send_io_message(dbg, 'code reload: reload finished\n')
                else:
                    _send_io_message(dbg, 'code reload: reload finished without applying any change\n')
        cmd = dbg.cmd_factory.make_reloaded_code_message(seq, found_module_to_reload)
        dbg.writer.add_command(cmd)
    except:
        pydev_log.exception('Error reloading code')