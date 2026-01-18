from tests.compat import mock
from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from tests.unit import MockServiceWithConfigTestCase
from boto.connection import AWSAuthConnection
from boto.s3.connection import S3Connection, HostRequiredError
from boto.s3.connection import S3ResponseError, Bucket
def test_non_retriable_host_returns_original_response(self):
    for code in self.retry_status_codes:
        with mock.patch.object(self.connection, '_mexe') as mocked_mexe:
            error_response = self.create_response(code)
            mocked_mexe.side_effect = [error_response]
            other_host = 'bucket.some-other-provider.com'
            response = self.connection.make_request('HEAD', '/', host=other_host)
            self.assertEqual(response, error_response)
            self.assertEqual(mocked_mexe.call_count, 1)
            self.assertEqual(mocked_mexe.call_args[0][0].host, other_host)