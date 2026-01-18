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
def is_parent_alive(self):
    if self.ppid != os.getppid():
        self.log.info('Parent changed, shutting down: %s', self)
        return False
    return True