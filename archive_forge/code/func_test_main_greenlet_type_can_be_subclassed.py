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
def test_main_greenlet_type_can_be_subclassed(self):
    main_type = self._check_current_is_main()
    subclass = type('subclass', (main_type,), {})
    self.assertIsNotNone(subclass)