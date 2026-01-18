import os
from oslotest import base
from oslo_privsep import priv_context
import oslo_privsep.tests
from oslo_privsep.tests import fixture
def assertNotMyPid(self, pid):
    self.assertIsInstance(pid, int)
    self.assertTrue(pid > 0)
    self.assertNotEqual(os.getpid(), pid)