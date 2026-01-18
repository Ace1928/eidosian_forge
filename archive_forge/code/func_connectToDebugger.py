import os
import sys
import traceback
from _pydev_bundle.pydev_imports import xmlrpclib, _queue, Exec
from  _pydev_bundle._pydev_calltip_util import get_description
from _pydevd_bundle import pydevd_vars
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle.pydevd_constants import (IS_JYTHON, NEXT_VALUE_SEPARATOR, get_global_debugger,
from contextlib import contextmanager
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_utils import interrupt_main_thread
from io import StringIO
def connectToDebugger(self, debuggerPort, debugger_options=None):
    """
        Used to show console with variables connection.
        Mainly, monkey-patches things in the debugger structure so that the debugger protocol works.
        """
    if debugger_options is None:
        debugger_options = {}
    env_key = 'PYDEVD_EXTRA_ENVS'
    if env_key in debugger_options:
        for env_name, value in debugger_options[env_key].items():
            existing_value = os.environ.get(env_name, None)
            if existing_value:
                os.environ[env_name] = '%s%c%s' % (existing_value, os.path.pathsep, value)
            else:
                os.environ[env_name] = value
            if env_name == 'PYTHONPATH':
                sys.path.append(value)
        del debugger_options[env_key]

    def do_connect_to_debugger():
        try:
            import pydevd
            from _pydev_bundle._pydev_saved_modules import threading
        except:
            traceback.print_exc()
            sys.stderr.write('pydevd is not available, cannot connect\n')
        from _pydevd_bundle.pydevd_constants import set_thread_id
        from _pydev_bundle import pydev_localhost
        set_thread_id(threading.current_thread(), 'console_main')
        VIRTUAL_FRAME_ID = '1'
        VIRTUAL_CONSOLE_ID = 'console_main'
        f = FakeFrame()
        f.f_back = None
        f.f_globals = {}
        f.f_locals = self.get_namespace()
        self.debugger = pydevd.PyDB()
        self.debugger.add_fake_frame(thread_id=VIRTUAL_CONSOLE_ID, frame_id=VIRTUAL_FRAME_ID, frame=f)
        try:
            pydevd.apply_debugger_options(debugger_options)
            self.debugger.connect(pydev_localhost.get_localhost(), debuggerPort)
            self.debugger.prepare_to_run()
            self.debugger.disable_tracing()
        except:
            traceback.print_exc()
            sys.stderr.write('Failed to connect to target debugger.\n')
        self.debugrunning = False
        try:
            import pydevconsole
            pydevconsole.set_debug_hook(self.debugger.process_internal_commands)
        except:
            traceback.print_exc()
            sys.stderr.write('Version of Python does not support debuggable Interactive Console.\n')
    self.exec_queue.put(do_connect_to_debugger)
    return ('connect complete',)