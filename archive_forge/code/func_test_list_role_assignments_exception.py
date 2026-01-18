import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_list_role_assignments_exception(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='role_assignments'), status_code=403)])
    with testtools.ExpectedException(exceptions.ForbiddenException):
        self.cloud.list_role_assignments()
    self.assert_calls()