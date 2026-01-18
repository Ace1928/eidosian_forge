import atexit
import errno
import logging
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from paste.deploy import loadapp, loadserver
from paste.script.command import Command, BadCommand
def restart_with_monitor(self, reloader=False):
    if self.verbose > 0:
        if reloader:
            print('Starting subprocess with file monitor')
        else:
            print('Starting subprocess with monitor parent')
    while 1:
        args = [self.quote_first_command_arg(sys.executable)] + sys.argv
        new_environ = os.environ.copy()
        if reloader:
            new_environ[self._reloader_environ_key] = 'true'
        else:
            new_environ[self._monitor_environ_key] = 'true'
        proc = None
        try:
            try:
                _turn_sigterm_into_systemexit()
                proc = subprocess.Popen(args, env=new_environ)
                exit_code = proc.wait()
                proc = None
            except KeyboardInterrupt:
                print('^C caught in monitor process')
                if self.verbose > 1:
                    raise
                return 1
        finally:
            if proc is not None and hasattr(os, 'kill'):
                import signal
                try:
                    os.kill(proc.pid, signal.SIGTERM)
                except (OSError, IOError):
                    pass
        if reloader:
            if exit_code != 3:
                return exit_code
        if self.verbose > 0:
            print('-' * 20, 'Restarting', '-' * 20)