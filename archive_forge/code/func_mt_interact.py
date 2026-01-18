import sys
import socket
import selectors
from time import monotonic as _time
import warnings
def mt_interact(self):
    """Multithreaded version of interact()."""
    import _thread
    _thread.start_new_thread(self.listener, ())
    while 1:
        line = sys.stdin.readline()
        if not line:
            break
        self.write(line.encode('ascii'))