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
def test_failed_to_switch_into_running(self):
    runs = []

    def func():
        runs.append(1)
        greenlet.getcurrent().parent.switch()
        runs.append(2)
        greenlet.getcurrent().parent.switch()
        runs.append(3)
    g = greenlet._greenlet.UnswitchableGreenlet(func)
    g.switch()
    self.assertEqual(runs, [1])
    g.switch()
    self.assertEqual(runs, [1, 2])
    g.force_switch_error = True
    with self.assertRaisesRegex(SystemError, 'Failed to switch stacks into a running greenlet.'):
        g.switch()
    g.force_switch_error = False
    g.switch()
    self.assertEqual(runs, [1, 2, 3])