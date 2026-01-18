import os
from typing import Dict, Mapping, Sequence
from hamcrest import assert_that, equal_to, not_
from hypothesis import given
from hypothesis.strategies import dictionaries, integers, lists
from twisted.python.systemd import ListenFDs
from twisted.trial.unittest import SynchronousTestCase
from .strategies import systemdDescriptorNames
def test_nonIntegerPIDVariable(self) -> None:
    """
        If the I{LISTEN_PID} environment variable is set to a string that cannot
        be parsed as an integer, no inherited descriptors are reported.
        """
    env = buildEnvironment(3, 'hello, world')
    sddaemon = ListenFDs.fromEnvironment(environ=env)
    self.assertEqual([], sddaemon.inheritedDescriptors())