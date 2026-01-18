from base64 import b64decode
from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
def test_get_signin_url_default(self):
    self.set_http_response(status_code=200)
    url = self.service_connection.get_signin_url()
    self.assertEqual(url, 'https://foocorporation.signin.aws.amazon.com/console/ec2')