from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional.identity.v3 import common
def test_group_set(self):
    group_name = self._create_dummy_group()
    new_group_name = data_utils.rand_name('NewTestGroup')
    raw_output = self.openstack('group set --domain %(domain)s --name %(new_group)s %(group)s' % {'domain': self.domain_name, 'new_group': new_group_name, 'group': group_name})
    self.assertEqual(0, len(raw_output))
    raw_output = self.openstack('group show --domain %(domain)s %(group)s' % {'domain': self.domain_name, 'group': new_group_name})
    group = self.parse_show_as_object(raw_output)
    self.assertEqual(new_group_name, group['name'])
    raw_output = self.openstack('group set --domain %(domain)s --name %(new_group)s %(group)s' % {'domain': self.domain_name, 'new_group': group_name, 'group': new_group_name})
    self.assertEqual(0, len(raw_output))