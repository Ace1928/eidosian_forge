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
def greenlet_in_thread_fn():
    VAR_VAR.set(1)
    is_running.set()
    should_suspend.wait(10)
    VAR_VAR.set(2)
    getcurrent().parent.switch()
    holder.append(VAR_VAR.get())