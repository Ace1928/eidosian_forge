import unittest
import inspect
import threading
def getArg(self, name):
    return self.args[name] if name in self.args else None