import uuid
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import roles
from testtools import matchers
def test_create_with_domain(self):
    ref = self.new_ref()
    domain_ref = self._new_domain_ref()
    domain_ref['id'] = uuid.uuid4().hex
    ref['domain_id'] = domain_ref['id']
    self.stub_entity('POST', entity=ref, status_code=201)
    returned = self.manager.create(name=ref['name'], domain=domain_ref)
    self.assertIsInstance(returned, self.model)
    for attr in ref:
        self.assertEqual(getattr(returned, attr), ref[attr], 'Expected different %s' % attr)