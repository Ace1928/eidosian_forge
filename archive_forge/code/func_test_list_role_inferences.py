import uuid
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import roles
from testtools import matchers
def test_list_role_inferences(self):
    prior_id = uuid.uuid4().hex
    prior_name = uuid.uuid4().hex
    implied_id = uuid.uuid4().hex
    implied_name = uuid.uuid4().hex
    mock_response = {'role_inferences': [{'implies': [{'id': implied_id, 'links': {'self': 'http://host/v3/roles/%s' % implied_id}, 'name': implied_name}], 'prior_role': {'id': prior_id, 'links': {'self': 'http://host/v3/roles/%s' % prior_id}, 'name': prior_name}}]}
    self.stub_url('GET', ['role_inferences'], json=mock_response, status_code=200)
    manager_result = self.manager.list_inference_roles()
    self.assertEqual(1, len(manager_result))
    self.assertIsInstance(manager_result[0], roles.InferenceRule)
    self.assertEqual(mock_response['role_inferences'][0]['implies'], manager_result[0].implies)
    self.assertEqual(mock_response['role_inferences'][0]['prior_role'], manager_result[0].prior_role)