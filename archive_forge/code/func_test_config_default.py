import os
from oslo_config import generator
import keystone.conf
from keystone.tests import unit
def test_config_default(self):
    self.assertIsNone(CONF.auth.password)
    self.assertIsNone(CONF.auth.token)
    self.assertEqual(False, CONF.profiler.enabled)