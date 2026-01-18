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
def test_failed_to_slp_switch_into_running(self):
    ex = self.assertScriptRaises('fail_slp_switch.py')
    self.assertIn('fail_slp_switch is running', ex.output)
    self.assertIn(ex.returncode, self.get_expected_returncodes_for_aborted_process())