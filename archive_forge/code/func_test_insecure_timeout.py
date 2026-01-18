import argparse
import uuid
from oslo_config import cfg
from oslo_config import fixture as config
from testtools import matchers
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
def test_insecure_timeout(self):
    s = self.get_session('--insecure --timeout 5.5')
    self.assertFalse(s.verify)
    self.assertEqual(5.5, s.timeout)