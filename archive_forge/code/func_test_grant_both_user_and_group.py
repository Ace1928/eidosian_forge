import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_grant_both_user_and_group(self):
    uris = self.__get('role', self.role_data, 'role_name', [], use_name=True)
    uris.extend(self.__user_mocks(self.user_data, use_name=True))
    uris.extend(self.__get('group', self.group_data, 'group_name', [], use_name=True))
    self.register_uris(uris)
    with testtools.ExpectedException(exceptions.SDKException, 'Specify either a group or a user, not both'):
        self.cloud.grant_role(self.role_data.role_name, user=self.user_data.name, group=self.group_data.group_name)
    self.assert_calls()