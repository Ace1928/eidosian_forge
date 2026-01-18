from base64 import b64decode
from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
def test_create_role_data_policy(self):
    self.set_http_response(status_code=200)
    self.service_connection.create_role('a_name', assume_role_policy_document={'hello': 'policy'})
    self.assert_request_parameters({'Action': 'CreateRole', 'AssumeRolePolicyDocument': '{"hello": "policy"}', 'RoleName': 'a_name'}, ignore_params_values=['Version'])