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
def test_falling_off_end_switches_to_unstarted_parent_works(self):

    def one_arg(x):
        return (x, 24)
    parent_never_started = RawGreenlet(one_arg)

    def leaf():
        return 42
    child = RawGreenlet(leaf, parent_never_started)
    result = child.switch()
    self.assertEqual(result, (42, 24))