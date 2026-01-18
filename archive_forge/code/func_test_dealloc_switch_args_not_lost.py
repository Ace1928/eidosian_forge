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
def test_dealloc_switch_args_not_lost(self):
    seen = []

    def worker():
        value = greenlet.getcurrent().parent.switch()
        del worker[0]
        initiator.parent = greenlet.getcurrent().parent
        try:
            greenlet.getcurrent().parent.switch(value)
        finally:
            seen.append(greenlet.getcurrent())

    def initiator():
        return 42
    worker = [RawGreenlet(worker)]
    worker[0].switch()
    initiator = RawGreenlet(initiator, worker[0])
    value = initiator.switch()
    self.assertTrue(seen)
    self.assertEqual(value, 42)