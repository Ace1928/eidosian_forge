from __future__ import annotations
import os
import sys
import types
from typing_extensions import NoReturn
from twisted.python import rebuild
from twisted.trial.unittest import TestCase
from . import crash_test_dummy
def test_hashException(self) -> None:
    """
        Rebuilding something that has a __hash__ that raises a non-TypeError
        shouldn't cause rebuild to die.
        """
    global unhashableObject
    unhashableObject = HashRaisesRuntimeError()

    def _cleanup() -> None:
        global unhashableObject
        unhashableObject = None
    self.addCleanup(_cleanup)
    rebuild.rebuild(rebuild)
    self.assertTrue(unhashableObject.hashCalled)