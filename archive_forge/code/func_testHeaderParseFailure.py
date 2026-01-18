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
@mock.patch(https_connection)
def testHeaderParseFailure(self, mock_conn):
    """Test incorrect header values do not raise exceptions."""
    mock_conn.putheader.return_value = None
    self.instance.putheader('content-encoding', 'gzip')
    self.instance.putheader('content-length', 'bytes 10')
    self.instance.putheader('content-range', 'not a number')
    self.assertAlmostEqual(self.instance.size_modifier, 1.0)