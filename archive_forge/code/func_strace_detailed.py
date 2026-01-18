import os
import signal
import subprocess
import tempfile
from . import errors
def strace_detailed(function, args, kwargs, follow_children=True):
    log_file = tempfile.NamedTemporaryFile()
    err_file = tempfile.NamedTemporaryFile()
    pid = os.getpid()
    strace_cmd = ['strace', '-r', '-tt', '-p', str(pid), '-o', log_file.name]
    if follow_children:
        strace_cmd.append('-f')
    proc = subprocess.Popen(strace_cmd, stdout=subprocess.PIPE, stderr=err_file.fileno())
    proc.stdout.readline()
    result = function(*args, **kwargs)
    os.kill(proc.pid, signal.SIGQUIT)
    proc.communicate()
    log_file.seek(0)
    log = log_file.read()
    log_file.close()
    err_file.seek(0)
    err_messages = err_file.read()
    err_file.close()
    if err_messages.startswith('attach: ptrace(PTRACE_ATTACH,'):
        raise StraceError(err_messages=err_messages)
    return (result, StraceResult(log, err_messages))