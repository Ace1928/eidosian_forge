from keystone.auth import schema
from keystone import exception
from keystone.tests import unit
def test_auth_no_identity_ex(self):
    self._expect_failure({})