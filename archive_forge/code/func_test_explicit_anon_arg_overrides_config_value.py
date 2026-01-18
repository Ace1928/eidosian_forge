from tests.compat import mock
from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from tests.unit import MockServiceWithConfigTestCase
from boto.connection import AWSAuthConnection
from boto.s3.connection import S3Connection, HostRequiredError
from boto.s3.connection import S3ResponseError, Bucket
def test_explicit_anon_arg_overrides_config_value(self):
    self.config = {'s3': {'no_sign_request': 'True'}}
    conn = self.connection_class(aws_access_key_id='less', aws_secret_access_key='more', host='s3.amazonaws.com', anon=False)
    url = conn.generate_url(0, 'GET', bucket='examplebucket', key='test.txt')
    self.assertIn('Signature=', url)