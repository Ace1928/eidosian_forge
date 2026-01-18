from base64 import b64decode
from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
def test_generate_credential_report(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.generate_credential_report()
    self.assertEquals(response['generate_credential_report_response']['generate_credential_report_result']['state'], 'COMPLETE')