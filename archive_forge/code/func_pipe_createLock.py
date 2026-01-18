import errno
import fcntl
import os
import eventlet
import eventlet.debug
import eventlet.greenthread
import eventlet.hubs
def pipe_createLock(self):
    """Replacement for logging.Handler.createLock method."""
    self.lock = PipeMutex()