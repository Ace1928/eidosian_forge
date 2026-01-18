from tests.unit import unittest
from boto.sts.connection import STSConnection
from tests.unit import AWSMockServiceTestCase
def test_assume_role_with_saml(self):
    arn = 'arn:aws:iam::000240903217:role/Test'
    principal = 'arn:aws:iam::000240903217:role/Principal'
    assertion = 'test'
    self.set_http_response(status_code=200)
    response = self.service_connection.assume_role_with_saml(role_arn=arn, principal_arn=principal, saml_assertion=assertion)
    self.assert_request_parameters({'RoleArn': arn, 'PrincipalArn': principal, 'SAMLAssertion': assertion, 'Action': 'AssumeRoleWithSAML'}, ignore_params_values=['Version'])
    self.assertEqual(response.credentials.access_key, 'accesskey')
    self.assertEqual(response.credentials.secret_key, 'secretkey')
    self.assertEqual(response.credentials.session_token, 'session_token')
    self.assertEqual(response.user.arn, 'arn:role')
    self.assertEqual(response.user.assume_role_id, 'roleid:myrolesession')