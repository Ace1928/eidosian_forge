from base64 import b64decode
from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
def test_get_credential_report(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.get_credential_report()
    b64decode(response['get_credential_report_response']['get_credential_report_result']['content'])