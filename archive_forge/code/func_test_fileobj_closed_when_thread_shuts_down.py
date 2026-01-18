import tempfile
from boto.compat import Queue
from tests.compat import mock, unittest
from tests.unit import AWSMockServiceTestCase
from boto.glacier.concurrent import ConcurrentUploader, ConcurrentDownloader
from boto.glacier.concurrent import UploadWorkerThread
from boto.glacier.concurrent import _END_SENTINEL
def test_fileobj_closed_when_thread_shuts_down(self):
    thread = UploadWorkerThread(mock.Mock(), 'vault_name', self.filename, 'upload_id', Queue(), Queue())
    fileobj = thread._fileobj
    self.assertFalse(fileobj.closed)
    thread.should_continue = False
    thread.run()
    self.assertTrue(fileobj.closed)