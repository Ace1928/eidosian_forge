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
def get_pid_file(server, pid_file):
    pid_file = os.path.abspath(pid_file) if pid_file else '/var/run/glance/%s.pid' % server
    dir, file = os.path.split(pid_file)
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except OSError:
            pass
    if not os.access(dir, os.W_OK):
        fallback = os.path.join(tempfile.mkdtemp(), '%s.pid' % server)
        msg = _('Unable to create pid file %(pid)s.  Running as non-root?\nFalling back to a temp file, you can stop %(service)s service using:\n  %(file)s %(server)s stop --pid-file %(fb)s') % {'pid': pid_file, 'service': server, 'file': __file__, 'server': server, 'fb': fallback}
        print(msg)
        pid_file = fallback
    return pid_file