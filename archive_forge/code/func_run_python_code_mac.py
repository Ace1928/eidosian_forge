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
def run_python_code_mac(pid, python_code, connect_debugger_tracing=False, show_debug_info=0):
    assert "'" not in python_code, 'Having a single quote messes with our command.'
    target_dll = get_target_filename()
    if not target_dll:
        raise RuntimeError('Could not find .dylib for attach to process.')
    libdir = os.path.dirname(__file__)
    lldb_prepare_file = find_helper_script(libdir, 'lldb_prepare.py')
    is_debug = 0
    cmd = ['lldb', '--no-lldbinit', '--script-language', 'Python']
    cmd.extend(["-o 'process attach --pid %d'" % pid, '-o \'command script import "%s"\'' % (lldb_prepare_file,), '-o \'load_lib_and_attach "%s" %s "%s" %s\'' % (target_dll, is_debug, python_code, show_debug_info)])
    cmd.extend(["-o 'process detach'", "-o 'script import os; os._exit(1)'"])
    env = os.environ.copy()
    env.pop('PYTHONIOENCODING', None)
    env.pop('PYTHONPATH', None)
    print('Running: %s' % ' '.join(cmd))
    subprocess.check_call(' '.join(cmd), shell=True, env=env)