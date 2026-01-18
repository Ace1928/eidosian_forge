from base64 import b64decode
from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
def test_create_role_default_cn_north(self):
    self.set_http_response(status_code=200)
    self.service_connection.host = 'iam.cn-north-1.amazonaws.com.cn'
    self.service_connection.create_role('a_name')
    self.assert_request_parameters({'Action': 'CreateRole', 'RoleName': 'a_name'}, ignore_params_values=['Version', 'AssumeRolePolicyDocument'])
    self.assertDictEqual(json.loads(self.actual_request.params['AssumeRolePolicyDocument']), {'Statement': [{'Action': ['sts:AssumeRole'], 'Effect': 'Allow', 'Principal': {'Service': ['ec2.amazonaws.com.cn']}}]})