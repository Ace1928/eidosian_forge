from unittest import mock
import uuid
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import users
def test_create_with_project_and_default_project(self):
    self.deprecations.expect_deprecations()
    ref = self.new_ref()
    self.stub_entity('POST', [self.collection_key], status_code=201, entity=ref)
    req_ref = ref.copy()
    req_ref.pop('id')
    param_ref = req_ref.copy()
    param_ref['project_id'] = 'project'
    params = utils.parameterize(param_ref)
    returned = self.manager.create(**params)
    self.assertIsInstance(returned, self.model)
    for attr in ref:
        self.assertEqual(getattr(returned, attr), ref[attr], 'Expected different %s' % attr)
    self.assertEntityRequestBodyIs(req_ref)