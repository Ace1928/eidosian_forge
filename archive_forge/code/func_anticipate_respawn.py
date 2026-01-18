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
@gated_by(CONF.respawn)
def anticipate_respawn(children):
    while children:
        pid, status = os.wait()
        if pid in children:
            pid_file, server, args = children.pop(pid)
            running = os.path.exists(pid_file)
            one_second_ago = time.time() - 1
            bouncing = running and os.path.getmtime(pid_file) >= one_second_ago
            if running and (not bouncing):
                args = (pid_file, server, args)
                new_pid = do_start('Respawn', *args)
                children[new_pid] = args
            else:
                rsn = 'bouncing' if bouncing else 'deliberately stopped'
                print(_('Suppressed respawn as %(serv)s was %(rsn)s.') % {'serv': server, 'rsn': rsn})