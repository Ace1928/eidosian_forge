from unittest import mock
from openstack.clustering.v1 import service
from openstack.tests.unit import base
def test_make_it(self):
    sot = service.Service(**EXAMPLE)
    self.assertEqual(EXAMPLE['host'], sot.host)
    self.assertEqual(EXAMPLE['binary'], sot.binary)
    self.assertEqual(EXAMPLE['status'], sot.status)
    self.assertEqual(EXAMPLE['state'], sot.state)
    self.assertEqual(EXAMPLE['disabled_reason'], sot.disabled_reason)
    self.assertEqual(EXAMPLE['updated_at'], sot.updated_at)