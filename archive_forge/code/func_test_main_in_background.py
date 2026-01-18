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
def test_main_in_background(self):
    main = greenlet.getcurrent()

    def run():
        return repr(main)
    g = RawGreenlet(run)
    r = g.switch()
    self.assertEndsWith(r, ' suspended active started main>')