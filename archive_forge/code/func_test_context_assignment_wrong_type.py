from __future__ import print_function
import gc
import sys
import unittest
from functools import partial
from unittest import skipUnless
from unittest import skipIf
from greenlet import greenlet
from greenlet import getcurrent
from . import TestCase
def test_context_assignment_wrong_type(self):
    g = greenlet()
    with self.assertRaisesRegex(TypeError, 'greenlet context must be a contextvars.Context or None'):
        g.gr_context = self