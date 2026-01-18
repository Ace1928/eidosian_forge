import os
import re
import sys
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_constants import get_global_debugger, IS_WINDOWS, IS_JYTHON, get_current_thread_id, \
from _pydev_bundle import pydev_log
from contextlib import contextmanager
from _pydevd_bundle import pydevd_constants, pydevd_defaults
from _pydevd_bundle.pydevd_defaults import PydevdCustomization
import ast
def new_execve(path, args, env):
    if _get_apply_arg_patching():
        args = patch_args(args, is_exec=True)
        send_process_created_message()
        send_process_about_to_be_replaced()
    return getattr(os, original_name)(path, args, env)