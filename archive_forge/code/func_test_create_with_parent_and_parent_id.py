import fixtures
import uuid
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneclient import exceptions as ksc_exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import projects
def test_create_with_parent_and_parent_id(self):
    ref = self._new_project_ref()
    ref['parent_id'] = uuid.uuid4().hex
    self.stub_entity('POST', entity=ref, status_code=201)
    returned = self.manager.create(name=ref['name'], domain=ref['domain_id'], parent=ref['parent_id'], parent_id=uuid.uuid4().hex)
    self.assertIsInstance(returned, self.model)
    for attr in ref:
        self.assertEqual(getattr(returned, attr), ref[attr], 'Expected different %s' % attr)
    self.assertEntityRequestBodyIs(ref)