import json
from tests.unit import AWSMockServiceTestCase
from boto.beanstalk.layer1 import Layer1
def test_create_application_version(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.create_application_version('application1', 'version1', s3_bucket='mybucket', s3_key='mykey', auto_create_application=True)
    app_version = api_response['CreateApplicationVersionResponse']['CreateApplicationVersionResult']['ApplicationVersion']
    self.assert_request_parameters({'Action': 'CreateApplicationVersion', 'ContentType': 'JSON', 'Version': '2010-12-01', 'ApplicationName': 'application1', 'AutoCreateApplication': 'true', 'SourceBundle.S3Bucket': 'mybucket', 'SourceBundle.S3Key': 'mykey', 'VersionLabel': 'version1'})
    self.assertEqual(app_version['ApplicationName'], 'application1')
    self.assertEqual(app_version['VersionLabel'], 'version1')