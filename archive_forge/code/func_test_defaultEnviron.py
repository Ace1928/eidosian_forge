import os
from typing import Dict, Mapping, Sequence
from hamcrest import assert_that, equal_to, not_
from hypothesis import given
from hypothesis.strategies import dictionaries, integers, lists
from twisted.python.systemd import ListenFDs
from twisted.trial.unittest import SynchronousTestCase
from .strategies import systemdDescriptorNames
def test_defaultEnviron(self) -> None:
    """
        If the process environment is not explicitly passed to
        L{ListenFDs.fromEnvironment}, the real process environment dictionary
        is used.
        """
    self.patch(os, 'environ', buildEnvironment(5, os.getpid()))
    sddaemon = ListenFDs.fromEnvironment()
    self.assertEqual(list(range(3, 3 + 5)), sddaemon.inheritedDescriptors())