import os
import sys
import eventlet
import __original_module_threading as orig_threading
import threading
import subprocess
import glance_store
from os_win import utilsfactory as os_win_utilsfactory
from oslo_config import cfg
from oslo_log import log as logging
from glance.common import config
from glance.common import exception
from glance import scrubber
def scrubber_already_running_posix():
    pid_file = '/var/run/glance/glance-scrubber.pid'
    if os.path.exists(os.path.abspath(pid_file)):
        return True
    for glance_scrubber_name in ['glance-scrubber', 'glance.cmd.scrubber']:
        cmd = subprocess.Popen(['/usr/bin/pgrep', '-f', glance_scrubber_name], stdout=subprocess.PIPE, shell=False)
        pids, _ = cmd.communicate()
        if isinstance(pids, bytes):
            pids = pids.decode()
        self_pid = os.getpid()
        if pids.count('\n') > 1 and str(self_pid) in pids:
            return True
        elif pids.count('\n') > 0 and str(self_pid) not in pids:
            return True
    return False