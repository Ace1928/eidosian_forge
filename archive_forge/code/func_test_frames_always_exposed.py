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
def test_frames_always_exposed(self):
    main = greenlet.getcurrent()

    def outer():
        inner(sys._getframe(0))

    def inner(frame):
        main.switch(frame)
    gr = RawGreenlet(outer)
    frame = gr.switch()
    unrelated = RawGreenlet(lambda: None)
    unrelated.switch()
    self.assertEqual(frame.f_code.co_name, 'outer')
    self.assertIsNone(frame.f_back)