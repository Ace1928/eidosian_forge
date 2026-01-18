import unittest
import inspect
import threading
def sendOverrider(self, data):
    self.lowerSink.append(data)