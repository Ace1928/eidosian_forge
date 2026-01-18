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
def handle_chld(self, sig, frame):
    """SIGCHLD handling"""
    self.reap_workers()
    self.wakeup()