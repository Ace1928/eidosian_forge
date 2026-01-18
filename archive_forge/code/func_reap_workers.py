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
def reap_workers(self):
    """        Reap workers to avoid zombie processes
        """
    try:
        while True:
            wpid, status = os.waitpid(-1, os.WNOHANG)
            if not wpid:
                break
            if self.reexec_pid == wpid:
                self.reexec_pid = 0
            else:
                exitcode = status >> 8
                if exitcode != 0:
                    self.log.error('Worker (pid:%s) exited with code %s', wpid, exitcode)
                if exitcode == self.WORKER_BOOT_ERROR:
                    reason = 'Worker failed to boot.'
                    raise HaltServer(reason, self.WORKER_BOOT_ERROR)
                if exitcode == self.APP_LOAD_ERROR:
                    reason = 'App failed to load.'
                    raise HaltServer(reason, self.APP_LOAD_ERROR)
                if exitcode > 0:
                    self.log.error('Worker (pid:%s) exited with code %s.', wpid, exitcode)
                elif status > 0:
                    try:
                        sig_name = signal.Signals(status).name
                    except ValueError:
                        sig_name = 'code {}'.format(status)
                    msg = 'Worker (pid:{}) was sent {}!'.format(wpid, sig_name)
                    if status == signal.SIGKILL:
                        msg += ' Perhaps out of memory?'
                    self.log.error(msg)
                worker = self.WORKERS.pop(wpid, None)
                if not worker:
                    continue
                worker.tmp.close()
                self.cfg.child_exit(self, worker)
    except OSError as e:
        if e.errno != errno.ECHILD:
            raise