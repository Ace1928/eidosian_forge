import argparse
import uuid
from oslo_config import cfg
from oslo_config import fixture as config
from testtools import matchers
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
def test_client_certs(self):
    cert = '/path/to/certfile'
    key = '/path/to/keyfile'
    s = self.get_session('--os-cert %s --os-key %s' % (cert, key))
    self.assertTrue(s.verify)
    self.assertEqual((cert, key), s.cert)