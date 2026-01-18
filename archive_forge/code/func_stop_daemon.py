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
def stop_daemon(self):
    pid_file = self.options.pid_file or 'paster.pid'
    if not os.path.exists(pid_file):
        print('No PID file exists in %s' % pid_file)
        return 1
    pid = read_pidfile(pid_file)
    if not pid:
        print('Not a valid PID file in %s' % pid_file)
        return 1
    pid = live_pidfile(pid_file)
    if not pid:
        print('PID in %s is not valid (deleting)' % pid_file)
        try:
            os.unlink(pid_file)
        except (OSError, IOError) as e:
            print('Could not delete: %s' % e)
            return 2
        return 1
    for j in range(10):
        if not live_pidfile(pid_file):
            break
        import signal
        os.kill(pid, signal.SIGINT)
        time.sleep(1)
    for j in range(10):
        if not live_pidfile(pid_file):
            break
        import signal
        os.kill(pid, signal.SIGTERM)
        time.sleep(1)
    else:
        print('failed to kill web process %s' % pid)
        return 3
    if os.path.exists(pid_file):
        os.unlink(pid_file)
    return 0