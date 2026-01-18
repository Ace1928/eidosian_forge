import tempfile
from boto.compat import Queue
from tests.compat import mock, unittest
from tests.unit import AWSMockServiceTestCase
from boto.glacier.concurrent import ConcurrentUploader, ConcurrentDownloader
from boto.glacier.concurrent import UploadWorkerThread
from boto.glacier.concurrent import _END_SENTINEL
def test_calculate_required_part_size(self):
    self.stat_mock.return_value.st_size = 1024 * 1024 * 8
    uploader = ConcurrentUploader(mock.Mock(), 'vault_name')
    total_parts, part_size = uploader._calculate_required_part_size(1024 * 1024 * 8)
    self.assertEqual(total_parts, 2)
    self.assertEqual(part_size, 4 * 1024 * 1024)