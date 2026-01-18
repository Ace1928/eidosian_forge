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
def test_dealloc(self):
    seen = []
    g1 = RawGreenlet(fmain)
    g2 = RawGreenlet(fmain)
    g1.switch(seen)
    g2.switch(seen)
    self.assertEqual(seen, [])
    del g1
    gc.collect()
    self.assertEqual(seen, [greenlet.GreenletExit])
    del g2
    gc.collect()
    self.assertEqual(seen, [greenlet.GreenletExit, greenlet.GreenletExit])