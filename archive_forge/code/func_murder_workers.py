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
def murder_workers(self):
    """        Kill unused/idle workers
        """
    if not self.timeout:
        return
    workers = list(self.WORKERS.items())
    for pid, worker in workers:
        try:
            if time.time() - worker.tmp.last_update() <= self.timeout:
                continue
        except (OSError, ValueError):
            continue
        if not worker.aborted:
            self.log.critical('WORKER TIMEOUT (pid:%s)', pid)
            worker.aborted = True
            self.kill_worker(pid, signal.SIGABRT)
        else:
            self.kill_worker(pid, signal.SIGKILL)