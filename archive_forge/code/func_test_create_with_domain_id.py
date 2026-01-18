import uuid
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import roles
from testtools import matchers
def test_create_with_domain_id(self):
    ref = self.new_ref()
    ref['domain_id'] = uuid.uuid4().hex
    self.test_create(ref=ref)