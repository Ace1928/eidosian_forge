from eventlet import hubs
from eventlet.support import greenlets as greenlet
def poll_result(self, notready=None):
    if self.has_result():
        return self.wait()
    return notready