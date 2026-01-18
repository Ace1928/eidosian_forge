from base64 import b64decode
from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
def test_get_signin_url_no_aliases(self):
    self.set_http_response(status_code=200)
    with self.assertRaises(Exception):
        self.service_connection.get_signin_url()