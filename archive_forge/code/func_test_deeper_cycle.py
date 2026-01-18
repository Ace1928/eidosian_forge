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
def test_deeper_cycle(self):
    g1 = RawGreenlet(lambda: None)
    g2 = RawGreenlet(lambda: None)
    g3 = RawGreenlet(lambda: None)
    g1.parent = g2
    g2.parent = g3
    with self.assertRaises(ValueError) as exc:
        g3.parent = g1
    self.assertEqual(str(exc.exception), 'cyclic parent chain')