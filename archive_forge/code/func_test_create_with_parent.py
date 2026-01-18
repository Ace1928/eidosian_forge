import fixtures
import uuid
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneclient import exceptions as ksc_exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import projects
def test_create_with_parent(self):
    parent_ref = self.new_ref()
    parent_ref['parent_id'] = uuid.uuid4().hex
    parent = self.test_create(ref=parent_ref)
    parent.id = parent_ref['id']
    ref = self.new_ref()
    ref['parent_id'] = parent.id
    child_ref = ref.copy()
    del child_ref['parent_id']
    child_ref['parent'] = parent
    del ref['id']
    self.stub_entity('GET', id=parent_ref['id'], entity=parent_ref)
    self.test_create(ref=child_ref, req_ref=ref)