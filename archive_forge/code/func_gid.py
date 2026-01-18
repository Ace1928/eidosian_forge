import os
import re
import signal as _signal
import sys
import time
import threading
import _thread
from cherrypy._cpcompat import text_or_bytes
from cherrypy._cpcompat import ntob
@gid.setter
def gid(self, val):
    if val is not None:
        if grp is None:
            self.bus.log('grp module not available; ignoring gid.', level=30)
            val = None
        elif isinstance(val, text_or_bytes):
            val = grp.getgrnam(val)[2]
    self._gid = val