from _pydev_bundle._pydev_saved_modules import thread, _code
from _pydevd_bundle.pydevd_constants import IS_JYTHON
from _pydevd_bundle.pydevconsole_code import InteractiveConsole
import os
import sys
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_constants import INTERACTIVE_MODE_AVAILABLE
import traceback
from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_save_locals
from _pydev_bundle.pydev_imports import Exec, _queue
import builtins as __builtin__
from _pydev_bundle.pydev_console_utils import BaseInterpreterInterface, BaseStdIn  # @UnusedImport
from _pydev_bundle.pydev_console_utils import CodeFragment
from _pydev_bundle.pydev_umd import runfile, _set_globals_function
def start_console_server(host, port, interpreter):
    try:
        if port == 0:
            host = ''
        from _pydev_bundle.pydev_imports import SimpleXMLRPCServer as XMLRPCServer
        try:
            server = XMLRPCServer((host, port), logRequests=False, allow_none=True)
        except:
            sys.stderr.write('Error starting server with host: "%s", port: "%s", client_port: "%s"\n' % (host, port, interpreter.client_port))
            sys.stderr.flush()
            raise
        _set_globals_function(interpreter.get_namespace)
        server.register_function(interpreter.execLine)
        server.register_function(interpreter.execMultipleLines)
        server.register_function(interpreter.getCompletions)
        server.register_function(interpreter.getFrame)
        server.register_function(interpreter.getVariable)
        server.register_function(interpreter.changeVariable)
        server.register_function(interpreter.getDescription)
        server.register_function(interpreter.close)
        server.register_function(interpreter.interrupt)
        server.register_function(interpreter.handshake)
        server.register_function(interpreter.connectToDebugger)
        server.register_function(interpreter.hello)
        server.register_function(interpreter.getArray)
        server.register_function(interpreter.evaluate)
        server.register_function(interpreter.ShowConsole)
        server.register_function(interpreter.loadFullValue)
        server.register_function(interpreter.enableGui)
        if port == 0:
            h, port = server.socket.getsockname()
            print(port)
            print(interpreter.client_port)
        while True:
            try:
                server.serve_forever()
            except:
                e = sys.exc_info()[1]
                retry = False
                try:
                    retry = e.args[0] == 4
                except:
                    pass
                if not retry:
                    raise
        return server
    except:
        pydev_log.exception()
        connection_queue = interpreter.get_connect_status_queue()
        if connection_queue is not None:
            connection_queue.put(False)