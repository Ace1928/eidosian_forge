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
def test_issue_245_reference_counting_subclass_no_threads(self):
    from greenlet import getcurrent
    from greenlet import GreenletExit

    class Greenlet(RawGreenlet):
        pass
    initial_refs = sys.getrefcount(Greenlet)
    self.glets = []

    def greenlet_main():
        try:
            getcurrent().parent.switch()
        except GreenletExit:
            self.glets.append(getcurrent())
    for _ in range(10):
        Greenlet(greenlet_main).switch()
    del self.glets
    self.assertEqual(sys.getrefcount(Greenlet), initial_refs)