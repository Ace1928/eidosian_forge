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
def test_cannot_delete_parent_of_main(self):
    with self.assertRaises(AttributeError) as exc:
        del greenlet.getcurrent().parent
    self.assertEqual(str(exc.exception), "can't delete attribute")