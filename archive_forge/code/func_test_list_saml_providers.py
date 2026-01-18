from base64 import b64decode
from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
def test_list_saml_providers(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.list_saml_providers()
    self.assert_request_parameters({'Action': 'ListSAMLProviders'}, ignore_params_values=['Version'])
    self.assertEqual(response.saml_provider_list, [{'arn': 'arn:aws:iam::123456789012:instance-profile/application_abc/component_xyz/Database', 'valid_until': '2032-05-09T16:27:11Z', 'create_date': '2012-05-09T16:27:03Z'}, {'arn': 'arn:aws:iam::123456789012:instance-profile/application_abc/component_xyz/Webserver', 'valid_until': '2015-03-11T13:11:02Z', 'create_date': '2012-05-09T16:27:11Z'}])