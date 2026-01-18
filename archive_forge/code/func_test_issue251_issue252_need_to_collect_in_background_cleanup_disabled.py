from __future__ import print_function, absolute_import, division
import sys
import gc
import time
import weakref
import threading
import greenlet
from . import TestCase
from .leakcheck import fails_leakcheck
from .leakcheck import ignores_leakcheck
from .leakcheck import RUNNING_ON_MANYLINUX
@fails_leakcheck
def test_issue251_issue252_need_to_collect_in_background_cleanup_disabled(self):
    self.expect_greenlet_leak = True
    greenlet._greenlet.enable_optional_cleanup(False)
    try:
        self._check_issue251(manually_collect_background=False)
    finally:
        greenlet._greenlet.enable_optional_cleanup(True)