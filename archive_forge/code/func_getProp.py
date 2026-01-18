import unittest
import inspect
import threading
def getProp(self, key, default=None):
    return self.getStack().getProp(key, default)