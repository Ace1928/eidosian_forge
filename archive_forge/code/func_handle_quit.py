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
def handle_quit(self, sig, frame):
    self.alive = False
    self.cfg.worker_int(self)
    self.tpool.shutdown(False)
    time.sleep(0.1)
    sys.exit(0)