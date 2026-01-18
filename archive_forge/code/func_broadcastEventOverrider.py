import unittest
import inspect
import threading
def broadcastEventOverrider(self, event):
    self.lowerEventSink.append(event)