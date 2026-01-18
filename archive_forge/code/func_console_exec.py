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
def console_exec(thread_id, frame_id, expression, dbg):
    """returns 'False' in case expression is partially correct
    """
    frame = dbg.find_frame(thread_id, frame_id)
    is_multiline = expression.count('@LINE@') > 1
    expression = str(expression.replace('@LINE@', '\n'))
    updated_globals = {}
    updated_globals.update(frame.f_globals)
    updated_globals.update(frame.f_locals)
    if IPYTHON:
        need_more = exec_code(CodeFragment(expression), updated_globals, frame.f_locals, dbg)
        if not need_more:
            pydevd_save_locals.save_locals(frame)
        return need_more
    interpreter = ConsoleWriter()
    if not is_multiline:
        try:
            code = compile_command(expression)
        except (OverflowError, SyntaxError, ValueError):
            interpreter.showsyntaxerror()
            return False
        if code is None:
            return True
    else:
        code = expression
    try:
        Exec(code, updated_globals, frame.f_locals)
    except SystemExit:
        raise
    except:
        interpreter.showtraceback()
    else:
        pydevd_save_locals.save_locals(frame)
    return False