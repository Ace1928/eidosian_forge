import tempfile
from boto.compat import Queue
from tests.compat import mock, unittest
from tests.unit import AWSMockServiceTestCase
from boto.glacier.concurrent import ConcurrentUploader, ConcurrentDownloader
from boto.glacier.concurrent import UploadWorkerThread
from boto.glacier.concurrent import _END_SENTINEL
def test_correct_low_level_api_calls(self):
    api_mock = mock.MagicMock()
    upload_id = '0898d645-ea45-4548-9a67-578f507ead49'
    initiate_upload_mock = mock.Mock(return_value={'UploadId': upload_id})
    api_mock.attach_mock(initiate_upload_mock, 'initiate_multipart_upload')
    uploader = FakeThreadedConcurrentUploader(api_mock, 'vault_name')
    uploader.upload('foofile')
    initiate_upload_mock.assert_called_with('vault_name', 4 * 1024 * 1024, None)
    api_mock.complete_multipart_upload.assert_called_with('vault_name', upload_id, mock.ANY, 8 * 1024 * 1024)