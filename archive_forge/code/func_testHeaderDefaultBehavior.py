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
def testHeaderDefaultBehavior(self, mock_conn):
    """Test the size modifier is correct under expected headers."""
    mock_conn.putheader.return_value = None
    self.instance.putheader('content-encoding', 'gzip')
    self.instance.putheader('content-length', '10')
    self.instance.putheader('content-range', 'bytes 0-104/*')
    self.assertAlmostEqual(self.instance.size_modifier, 10.5)