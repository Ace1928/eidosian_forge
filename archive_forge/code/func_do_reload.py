import argparse
import fcntl
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from oslo_config import cfg
from oslo_utils import units
from glance.common import config
from glance.i18n import _
def do_reload(pid_file, server):
    if server not in RELOAD_SERVERS:
        msg = _('Reload of %(serv)s not supported') % {'serv': server}
        sys.exit(msg)
    pid = None
    if os.path.exists(pid_file):
        with open(pid_file, 'r') as pidfile:
            pid = int(pidfile.read().strip())
    else:
        msg = _('Server %(serv)s is stopped') % {'serv': server}
        sys.exit(msg)
    sig = signal.SIGHUP
    try:
        print(_('Reloading %(serv)s (pid %(pid)s) with signal(%(sig)s)') % {'serv': server, 'pid': pid, 'sig': sig})
        os.kill(pid, sig)
    except OSError:
        print(_('Process %d not running') % pid)