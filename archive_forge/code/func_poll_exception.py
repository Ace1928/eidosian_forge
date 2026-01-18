from eventlet import hubs
from eventlet.support import greenlets as greenlet
def poll_exception(self, notready=None):
    if self.has_exception():
        return self.wait()
    return notready