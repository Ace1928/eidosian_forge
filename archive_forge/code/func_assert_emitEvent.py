import unittest
import inspect
import threading
def assert_emitEvent(self, event):
    self.emitEvent(event)
    try:
        self.assertEqual(event, self.upperEventSink.pop())
    except IndexError:
        raise AssertionError("Event '%s' was not emited through this layer" % event.getName())