from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from openstackclient.tests.functional.identity.v3 import common
def test_domain_multi_delete(self):
    domain_1 = self._create_dummy_domain(add_clean_up=False)
    domain_2 = self._create_dummy_domain(add_clean_up=False)
    raw_output = self.openstack('domain set --disable %s' % domain_1)
    self.assertEqual(0, len(raw_output))
    raw_output = self.openstack('domain set --disable %s' % domain_2)
    self.assertEqual(0, len(raw_output))
    raw_output = self.openstack('domain delete %s %s' % (domain_1, domain_2))
    self.assertEqual(0, len(raw_output))