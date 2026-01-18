from __future__ import annotations
import os
import sys
import types
from typing_extensions import NoReturn
from twisted.python import rebuild
from twisted.trial.unittest import TestCase
from . import crash_test_dummy
def test_Rebuild(self) -> None:
    """
        Rebuilding an unchanged module.
        """
    x = crash_test_dummy.X('a')
    rebuild.rebuild(crash_test_dummy, doLog=False)
    x.do()
    self.assertEqual(x.__class__, crash_test_dummy.X)
    self.assertEqual(f, crash_test_dummy.foo)