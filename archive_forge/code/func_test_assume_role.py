from tests.unit import unittest
from boto.sts.connection import STSConnection
from tests.unit import AWSMockServiceTestCase
def test_assume_role(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.assume_role('arn:role', 'mysession')
    self.assert_request_parameters({'Action': 'AssumeRole', 'RoleArn': 'arn:role', 'RoleSessionName': 'mysession'}, ignore_params_values=['Version'])
    self.assertEqual(response.credentials.access_key, 'accesskey')
    self.assertEqual(response.credentials.secret_key, 'secretkey')
    self.assertEqual(response.credentials.session_token, 'session_token')
    self.assertEqual(response.user.arn, 'arn:role')
    self.assertEqual(response.user.assume_role_id, 'roleid:myrolesession')