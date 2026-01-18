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
def test_dealloc_catches_GreenletExit_throws_other(self):

    def run():
        try:
            greenlet.getcurrent().parent.switch()
        except greenlet.GreenletExit:
            raise SomeError from None
    g = RawGreenlet(run)
    g.switch()
    oldstderr = sys.stderr
    try:
        from cStringIO import StringIO
    except ImportError:
        from io import StringIO
    stderr = sys.stderr = StringIO()
    try:
        del g
    finally:
        sys.stderr = oldstderr
    v = stderr.getvalue()
    self.assertIn('Exception', v)
    self.assertIn('ignored', v)
    self.assertIn('SomeError', v)