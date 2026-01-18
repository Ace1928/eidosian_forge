from keystone.auth import schema
from keystone import exception
from keystone.tests import unit
def test_identity_not_object_ex(self):
    self._expect_failure({'identity': 'something'})