import io
from tests.compat import mock, unittest
from tests.unit import AWSMockServiceTestCase
from boto.compat import StringIO
from boto.exception import BotoServerError
from boto.exception import ResumableDownloadException
from boto.exception import ResumableTransferDisposition
from boto.s3.connection import S3Connection
from boto.s3.bucket import Bucket
from boto.s3.key import Key
def test_download_raises_retriable_error_with_truncated_stream(self):
    test_case_headers = [[('Content-Length', '5')], [('Content-Range', 'bytes 15-19/100')]]
    for headers in test_case_headers:
        with self.subTest(headers=headers):
            head_object_response = self.create_response(status_code=200, header=headers)
            media_response = self.create_response(status_code=200, header=headers)
            media_response.read.side_effect = [b'1234', b'']
            self.https_connection.getresponse.side_effect = [head_object_response, media_response]
            bucket = Bucket(self.service_connection, 'bucket')
            key = bucket.get_key('object')
            with self.assertRaisesRegex(ResumableDownloadException, 'Download stream truncated. Received 4 of 5 bytes.') as context:
                key.get_file(io.BytesIO())
                self.assertEqual(context.exception.disposition, ResumableTransferDisposition.WAIT_BEFORE_RETRY)