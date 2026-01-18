from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
def test_get_policy(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.get_policy('arn:aws:iam::123456789012:policy/S3-read-only-example-bucket')
    self.assert_request_parameters({'Action': 'GetPolicy', 'PolicyArn': 'arn:aws:iam::123456789012:policy/S3-read-only-example-bucket'}, ignore_params_values=['Version'])
    self.assertEqual(response['get_policy_response']['get_policy_result']['policy']['arn'], 'arn:aws:iam::123456789012:policy/S3-read-only-example-bucket')
    self.assertEqual(response['get_policy_response']['get_policy_result']['policy']['description'], 'My Awesome Policy')