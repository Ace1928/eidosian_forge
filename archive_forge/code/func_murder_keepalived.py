from concurrent import futures
import errno
import os
import selectors
import socket
import ssl
import sys
import time
from collections import deque
from datetime import datetime
from functools import partial
from threading import RLock
from . import base
from .. import http
from .. import util
from .. import sock
from ..http import wsgi
def murder_keepalived(self):
    now = time.time()
    while True:
        with self._lock:
            try:
                conn = self._keep.popleft()
            except IndexError:
                break
        delta = conn.timeout - now
        if delta > 0:
            with self._lock:
                self._keep.appendleft(conn)
            break
        else:
            self.nr_conns -= 1
            with self._lock:
                try:
                    self.poller.unregister(conn.sock)
                except EnvironmentError as e:
                    if e.errno != errno.EBADF:
                        raise
                except KeyError:
                    pass
                except ValueError:
                    pass
            conn.close()