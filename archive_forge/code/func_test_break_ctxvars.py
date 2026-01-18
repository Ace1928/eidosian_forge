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
def test_break_ctxvars(self):
    let1 = greenlet(copy_context().run)
    let2 = greenlet(copy_context().run)
    let1.switch(getcurrent().switch)
    let2.switch(getcurrent().switch)
    let1.switch()