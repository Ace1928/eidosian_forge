from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional.identity.v3 import common
def test_group_list_with_domain(self):
    group_name = self._create_dummy_group()
    raw_output = self.openstack('group list --domain %s' % self.domain_name)
    items = self.parse_listing(raw_output)
    self.assert_table_structure(items, common.BASIC_LIST_HEADERS)
    self.assertIn(group_name, raw_output)