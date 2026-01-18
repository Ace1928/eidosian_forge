import os
from typing import Dict, Mapping, Sequence
from hamcrest import assert_that, equal_to, not_
from hypothesis import given
from hypothesis.strategies import dictionaries, integers, lists
from twisted.python.systemd import ListenFDs
from twisted.trial.unittest import SynchronousTestCase
from .strategies import systemdDescriptorNames
def test_missingPIDVariable(self) -> None:
    """
        If the I{LISTEN_PID} environment variable is not present then
        there is no clear indication that any file descriptors were inherited
        by this process so L{ListenFDs.inheritedDescriptors} returns an empty
        list.
        """
    env = buildEnvironment(3, os.getpid())
    del env['LISTEN_PID']
    sddaemon = ListenFDs.fromEnvironment(environ=env)
    self.assertEqual([], sddaemon.inheritedDescriptors())