import unittest
import inspect
import threading
def receiveOverrider(self, data):
    self.upperSink.append(data)