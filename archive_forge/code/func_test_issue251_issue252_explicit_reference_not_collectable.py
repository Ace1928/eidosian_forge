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
def test_issue251_issue252_explicit_reference_not_collectable(self):
    self._check_issue251(manually_collect_background=False, explicit_reference_to_switch=True)