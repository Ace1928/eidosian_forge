from unittest import mock
from keystoneauth1 import adapter
from openstack.identity.v3 import domain
from openstack.identity.v3 import group
from openstack.identity.v3 import role
from openstack.identity.v3 import user
from openstack.tests.unit import base
def test_assign_role_to_user_bad(self):
    sot = domain.Domain(**EXAMPLE)
    resp = self.bad_resp
    self.sess.put = mock.Mock(return_value=resp)
    self.assertFalse(sot.assign_role_to_user(self.sess, user.User(id='1'), role.Role(id='2')))