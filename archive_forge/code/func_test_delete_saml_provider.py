from base64 import b64decode
from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
def test_delete_saml_provider(self):
    self.set_http_response(status_code=200)
    self.service_connection.delete_saml_provider('arn')
    self.assert_request_parameters({'Action': 'DeleteSAMLProvider', 'SAMLProviderArn': 'arn'}, ignore_params_values=['Version'])