from collections import deque
from contextlib import contextmanager
import sys
import time
from eventlet.pools import Pool
from eventlet import timeout
from eventlet import hubs
from eventlet.hubs.timer import Timer
from eventlet.greenthread import GreenThread
def credentials_for(self, host):
    if host in self._credentials:
        return self._credentials[host]
    else:
        return self._credentials.get('default', None)