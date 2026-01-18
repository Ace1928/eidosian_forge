import ctypes
import os
import struct
import subprocess
import sys
import time
from contextlib import contextmanager
import platform
import traceback
import os, time, sys
def run_python_code_linux(pid, python_code, connect_debugger_tracing=False, show_debug_info=0):
    assert "'" not in python_code, 'Having a single quote messes with our command.'
    target_dll = get_target_filename()
    if not target_dll:
        raise RuntimeError('Could not find .so for attach to process.')
    target_dll_name = os.path.splitext(os.path.basename(target_dll))[0]
    is_debug = 0
    cmd = ['gdb', '--nw', '--nh', '--nx', '--pid', str(pid), '--batch']
    gdb_load_shared_libraries = os.environ.get('PYDEVD_GDB_SCAN_SHARED_LIBRARIES', '').strip()
    if gdb_load_shared_libraries:
        print('PYDEVD_GDB_SCAN_SHARED_LIBRARIES set: %s.' % (gdb_load_shared_libraries,))
        cmd.extend(["--init-eval-command='set auto-solib-add off'"])
        for lib in gdb_load_shared_libraries.split(','):
            lib = lib.strip()
            cmd.extend(["--eval-command='sharedlibrary %s'" % (lib,)])
    else:
        print('PYDEVD_GDB_SCAN_SHARED_LIBRARIES not set (scanning all libraries for needed symbols).')
    cmd.extend(["--eval-command='set scheduler-locking off'"])
    cmd.extend(["--eval-command='set architecture auto'"])
    cmd.extend(['--eval-command=\'call (void*)dlopen("%s", 2)\'' % target_dll, "--eval-command='sharedlibrary %s'" % target_dll_name, '--eval-command=\'call (int)DoAttach(%s, "%s", %s)\'' % (is_debug, python_code, show_debug_info)])
    env = os.environ.copy()
    env.pop('PYTHONIOENCODING', None)
    env.pop('PYTHONPATH', None)
    print('Running: %s' % ' '.join(cmd))
    subprocess.check_call(' '.join(cmd), shell=True, env=env)