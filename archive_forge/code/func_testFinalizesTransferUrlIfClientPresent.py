import string
import unittest
import httplib2
import json
import mock
import six
from six.moves import http_client
from apitools.base.py import base_api
from apitools.base.py import exceptions
from apitools.base.py import gzip
from apitools.base.py import http_wrapper
from apitools.base.py import transfer
@mock.patch.object(transfer.Upload, 'RefreshResumableUploadState', new=mock.Mock())
def testFinalizesTransferUrlIfClientPresent(self):
    """Tests upload's enforcement of client custom endpoints."""
    mock_client = mock.Mock()
    mock_http = mock.Mock()
    fake_json_data = json.dumps({'auto_transfer': False, 'mime_type': '', 'total_size': 0, 'url': 'url'})
    transfer.Upload.FromData(self.sample_stream, fake_json_data, mock_http, client=mock_client)
    mock_client.FinalizeTransferUrl.assert_called_once_with('url')