from datetime import datetime
import errno
import os
import select
import socket
import ssl
import sys
from gunicorn import http
from gunicorn.http import wsgi
from gunicorn import sock
from gunicorn import util
from gunicorn.workers import base
def run_for_multiple(self, timeout):
    while self.alive:
        self.notify()
        try:
            ready = self.wait(timeout)
        except StopWaiting:
            return
        if ready is not None:
            for listener in ready:
                if listener == self.PIPE[0]:
                    continue
                try:
                    self.accept(listener)
                except EnvironmentError as e:
                    if e.errno not in (errno.EAGAIN, errno.ECONNABORTED, errno.EWOULDBLOCK):
                        raise
        if not self.is_parent_alive():
            return