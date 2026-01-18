from unittest import mock
import uuid
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import users
def test_create_doesnt_log_password(self):
    password = uuid.uuid4().hex
    ref = self.new_ref()
    self.stub_entity('POST', [self.collection_key], status_code=201, entity=ref)
    req_ref = ref.copy()
    req_ref.pop('id')
    param_ref = req_ref.copy()
    param_ref['password'] = password
    params = utils.parameterize(param_ref)
    self.manager.create(**params)
    self.assertNotIn(password, self.logger.output)