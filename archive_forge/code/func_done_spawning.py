import traceback
import eventlet
from eventlet import queue
from eventlet.support import greenlets as greenlet
def done_spawning(self):
    self.spawn(lambda: StopIteration())