from __future__ import annotations
import os
import sys
import types
from typing_extensions import NoReturn
from twisted.python import rebuild
from twisted.trial.unittest import TestCase
from . import crash_test_dummy
def test_ComponentInteraction(self) -> None:
    x = crash_test_dummy.XComponent()
    x.setAdapter(crash_test_dummy.IX, crash_test_dummy.XA)
    x.getComponent(crash_test_dummy.IX)
    rebuild.rebuild(crash_test_dummy, 0)
    newComponent = x.getComponent(crash_test_dummy.IX)
    newComponent.method()
    self.assertEqual(newComponent.__class__, crash_test_dummy.XA)
    from twisted.python import components
    self.assertRaises(ValueError, components.registerAdapter, crash_test_dummy.XA, crash_test_dummy.X, crash_test_dummy.IX)