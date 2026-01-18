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
def internal_get_smart_step_into_variants(dbg, seq, thread_id, frame_id, start_line, end_line, set_additional_thread_info):
    try:
        thread = pydevd_find_thread_by_id(thread_id)
        frame = dbg.find_frame(thread_id, frame_id)
        if thread is None or frame is None:
            cmd = dbg.cmd_factory.make_error_message(seq, 'Frame not found: %s from thread: %s' % (frame_id, thread_id))
            dbg.writer.add_command(cmd)
            return
        if pydevd_bytecode_utils is None:
            variants = []
        else:
            variants = pydevd_bytecode_utils.calculate_smart_step_into_variants(frame, int(start_line), int(end_line))
        info = set_additional_thread_info(thread)
        info.pydev_smart_step_into_variants = tuple(variants)
        xml = '<xml>'
        for variant in variants:
            if variant.children_variants:
                for child_variant in variant.children_variants:
                    xml += '<variant name="%s" isVisited="%s" line="%s" offset="%s" childOffset="%s" callOrder="%s"/>' % (quote(child_variant.name), str(child_variant.is_visited).lower(), child_variant.line, variant.offset, child_variant.offset, child_variant.call_order)
            else:
                xml += '<variant name="%s" isVisited="%s" line="%s" offset="%s" childOffset="-1" callOrder="%s"/>' % (quote(variant.name), str(variant.is_visited).lower(), variant.line, variant.offset, variant.call_order)
        xml += '</xml>'
        cmd = NetCommand(CMD_GET_SMART_STEP_INTO_VARIANTS, seq, xml)
        dbg.writer.add_command(cmd)
    except:
        pydev_log.exception('Error calculating Smart Step Into Variants.')
        cmd = dbg.cmd_factory.make_error_message(seq, 'Error getting smart step into variants for frame: %s from thread: %s' % (frame_id, thread_id))
        dbg.writer.add_command(cmd)