import os
import re
import signal as _signal
import sys
import time
import threading
import _thread
from cherrypy._cpcompat import text_or_bytes
from cherrypy._cpcompat import ntob
def release_thread(self):
    """Release the current thread and run 'stop_thread' listeners."""
    thread_ident = _thread.get_ident()
    i = self.threads.pop(thread_ident, None)
    if i is not None:
        self.bus.publish('stop_thread', i)