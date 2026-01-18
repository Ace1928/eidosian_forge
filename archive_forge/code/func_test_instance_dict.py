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
def test_instance_dict(self):

    def f():
        greenlet.getcurrent().test = 42

    def deldict(g):
        del g.__dict__

    def setdict(g, value):
        g.__dict__ = value
    g = RawGreenlet(f)
    self.assertEqual(g.__dict__, {})
    g.switch()
    self.assertEqual(g.test, 42)
    self.assertEqual(g.__dict__, {'test': 42})
    g.__dict__ = g.__dict__
    self.assertEqual(g.__dict__, {'test': 42})
    self.assertRaises(TypeError, deldict, g)
    self.assertRaises(TypeError, setdict, g, 42)