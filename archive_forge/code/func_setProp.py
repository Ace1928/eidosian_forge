import unittest
import inspect
import threading
def setProp(self, key, val):
    return self.getStack().setProp(key, val)