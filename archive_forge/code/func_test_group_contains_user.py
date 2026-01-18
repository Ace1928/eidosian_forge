from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional.identity.v3 import common
def test_group_contains_user(self):
    group_name = self._create_dummy_group()
    username = self._create_dummy_user()
    raw_output = self.openstack('group add user --group-domain %(group_domain)s --user-domain %(user_domain)s %(group)s %(user)s' % {'group_domain': self.domain_name, 'user_domain': self.domain_name, 'group': group_name, 'user': username})
    self.addCleanup(self.openstack, 'group remove user --group-domain %(group_domain)s --user-domain %(user_domain)s %(group)s %(user)s' % {'group_domain': self.domain_name, 'user_domain': self.domain_name, 'group': group_name, 'user': username})
    self.assertOutput('', raw_output)
    raw_output = self.openstack('group contains user --group-domain %(group_domain)s --user-domain %(user_domain)s %(group)s %(user)s' % {'group_domain': self.domain_name, 'user_domain': self.domain_name, 'group': group_name, 'user': username})
    self.assertEqual('%(user)s in group %(group)s\n' % {'user': username, 'group': group_name}, raw_output)