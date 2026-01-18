import sys
import queue
import threading
from pyglet import app
from pyglet import clock
from pyglet import event
def dispatch_posted_events(self):
    """Immediately dispatch all pending events.

        Normally this is called automatically by the runloop iteration.
        """
    while True:
        try:
            dispatcher, evnt, args = self._event_queue.get(False)
            dispatcher.dispatch_event(evnt, *args)
        except queue.Empty:
            break
        except ReferenceError:
            pass