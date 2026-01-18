import errno
import os
import random
import select
import signal
import sys
import time
import traceback
from gunicorn.errors import HaltServer, AppImportError
from gunicorn.pidfile import Pidfile
from gunicorn import sock, systemd, util
from gunicorn import __version__, SERVER_SOFTWARE
def maybe_promote_master(self):
    if self.master_pid == 0:
        return
    if self.master_pid != os.getppid():
        self.log.info('Master has been promoted.')
        self.master_name = 'Master'
        self.master_pid = 0
        self.proc_name = self.cfg.proc_name
        del os.environ['GUNICORN_PID']
        if self.pidfile is not None:
            self.pidfile.rename(self.cfg.pidfile)
        util._setproctitle('master [%s]' % self.proc_name)