import unittest
import inspect
import threading
def getLayerInterface(self, YowLayerClass):
    for s in self.sublayers:
        if s.__class__ == YowLayerClass:
            return s.getLayerInterface()