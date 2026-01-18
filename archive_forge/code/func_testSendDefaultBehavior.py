from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import email
from gslib.gcs_json_media import BytesTransferredContainer
from gslib.gcs_json_media import HttpWithDownloadStream
from gslib.gcs_json_media import UploadCallbackConnectionClassFactory
import gslib.tests.testcase as testcase
import httplib2
import io
import six
from six import add_move, MovedModule
from six.moves import http_client
from six.moves import mock
@mock.patch('gslib.progress_callback.ProgressCallbackWithTimeout')
@mock.patch('httplib2.HTTPSConnectionWithTimeout')
def testSendDefaultBehavior(self, mock_conn, mock_callback):
    mock_conn.send.return_value = None
    self.instance.size_modifier = 2
    self.instance.processed_initial_bytes = True
    self.instance.callback_processor = mock_callback
    sample_data = b'0123456789'
    self.instance.send(sample_data)
    self.assertTrue(mock_conn.send.called)
    (_, sent_data), _ = mock_conn.send.call_args_list[0]
    self.assertEqual(sent_data, sample_data)
    self.assertTrue(mock_callback.Progress.called)
    [sent_bytes], _ = mock_callback.Progress.call_args_list[0]
    self.assertEqual(sent_bytes, 20)