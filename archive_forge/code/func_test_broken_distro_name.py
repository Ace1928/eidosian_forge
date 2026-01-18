import os
import os.path
import stat
import unittest
from fixtures import MockPatch, TempDir
from testtools import TestCase
from lazr.restfulclient.authorize.oauth import (
def test_broken_distro_name(self):
    self.useFixture(MockPatch('distro.name', side_effect=Exception('Oh noes!')))
    self.useFixture(MockPatch('platform.system', return_value='BazOS'))
    self.useFixture(MockPatch('socket.gethostname', return_value='baz'))
    consumer = SystemWideConsumer('app name')
    self.assertEqual(consumer.key, 'System-wide: BazOS (baz)')