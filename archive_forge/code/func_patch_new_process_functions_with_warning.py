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
def patch_new_process_functions_with_warning():
    monkey_patch_os('execl', create_warn_multiproc)
    monkey_patch_os('execle', create_warn_multiproc)
    monkey_patch_os('execlp', create_warn_multiproc)
    monkey_patch_os('execlpe', create_warn_multiproc)
    monkey_patch_os('execv', create_warn_multiproc)
    monkey_patch_os('execve', create_warn_multiproc)
    monkey_patch_os('execvp', create_warn_multiproc)
    monkey_patch_os('execvpe', create_warn_multiproc)
    monkey_patch_os('spawnl', create_warn_multiproc)
    monkey_patch_os('spawnle', create_warn_multiproc)
    monkey_patch_os('spawnlp', create_warn_multiproc)
    monkey_patch_os('spawnlpe', create_warn_multiproc)
    monkey_patch_os('spawnv', create_warn_multiproc)
    monkey_patch_os('spawnve', create_warn_multiproc)
    monkey_patch_os('spawnvp', create_warn_multiproc)
    monkey_patch_os('spawnvpe', create_warn_multiproc)
    monkey_patch_os('posix_spawn', create_warn_multiproc)
    if not IS_JYTHON:
        if not IS_WINDOWS:
            monkey_patch_os('fork', create_warn_multiproc)
            try:
                import _posixsubprocess
                monkey_patch_module(_posixsubprocess, 'fork_exec', create_warn_fork_exec)
            except ImportError:
                pass
            try:
                import subprocess
                monkey_patch_module(subprocess, '_fork_exec', create_subprocess_warn_fork_exec)
            except AttributeError:
                pass
        else:
            try:
                import _subprocess
            except ImportError:
                import _winapi as _subprocess
            monkey_patch_module(_subprocess, 'CreateProcess', create_CreateProcessWarnMultiproc)