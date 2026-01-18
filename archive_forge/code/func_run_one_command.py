import functools
import logging
from multiprocessing import managers
import os
import shutil
import signal
import stat
import sys
import tempfile
import threading
import time
from oslo_rootwrap import cmd
from oslo_rootwrap import jsonrpc
from oslo_rootwrap import subprocess
from oslo_rootwrap import wrapper
def run_one_command(self, userargs, stdin=None):
    self.reset_timer()
    try:
        obj = wrapper.start_subprocess(self.filters, userargs, exec_dirs=self.config.exec_dirs, log=self.config.use_syslog, close_fds=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except wrapper.FilterMatchNotExecutable:
        LOG.warning('Executable not found for: %s', ' '.join(userargs))
        return (cmd.RC_NOEXECFOUND, '', '')
    except wrapper.NoFilterMatched:
        LOG.warning('Unauthorized command: %s (no filter matched)', ' '.join(userargs))
        return (cmd.RC_UNAUTHORIZED, '', '')
    if stdin is not None:
        stdin = os.fsencode(stdin)
    out, err = obj.communicate(stdin)
    out = os.fsdecode(out)
    err = os.fsdecode(err)
    return (obj.returncode, out, err)