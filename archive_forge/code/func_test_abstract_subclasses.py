from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gc
import sys
import time
import threading
from abc import ABCMeta, abstractmethod
import greenlet
from greenlet import greenlet as RawGreenlet
from . import TestCase
from .leakcheck import fails_leakcheck
def test_abstract_subclasses(self):
    AbstractSubclass = ABCMeta('AbstractSubclass', (RawGreenlet,), {'run': abstractmethod(lambda self: None)})

    class BadSubclass(AbstractSubclass):
        pass

    class GoodSubclass(AbstractSubclass):

        def run(self):
            pass
    GoodSubclass()
    self.assertRaises(TypeError, BadSubclass)