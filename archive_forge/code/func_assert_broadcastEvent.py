import unittest
import inspect
import threading
def assert_broadcastEvent(self, event):
    self.broadcastEvent(event)
    try:
        self.assertEqual(event, self.lowerEventSink.pop())
    except IndexError:
        raise AssertionError("Event '%s' was not broadcasted through this layer" % event.getName())