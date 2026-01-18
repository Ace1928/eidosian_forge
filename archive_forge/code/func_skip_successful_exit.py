import os
import sys
import traceback
from pydevconsole import InterpreterInterface, process_exec_queue, start_console_server, init_mpl_in_console
from _pydev_bundle._pydev_saved_modules import threading, _queue
from _pydev_bundle import pydev_imports
from _pydevd_bundle.pydevd_utils import save_main_module
from _pydev_bundle.pydev_console_utils import StdIn
from pydevd_file_utils import get_fullname
def skip_successful_exit(*args):
    """ System exit in file shouldn't kill interpreter (i.e. in `timeit`)"""
    if len(args) == 1 and args[0] in (0, None):
        pass
    else:
        raise SystemExit(*args)