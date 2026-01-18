import json
import os.path
import shutil
import tempfile
import unittest
import mock
import six
from apitools.base.py import credentials_lib
from apitools.base.py import util
@mock.patch.object(util, 'DetectGce', autospec=True)
def testGceServiceAccountsCached(self, mock_detect):
    mock_detect.return_value = True
    tempd = tempfile.mkdtemp()
    tempname = os.path.join(tempd, 'creds')
    scopes = ['scope1']
    service_account_name = 'some_service_account_name'
    metadatamock = MetadataMock(scopes, service_account_name)
    with mock.patch.object(credentials_lib, '_GceMetadataRequest', side_effect=metadatamock, autospec=True) as opener_mock:
        try:
            creds1 = self._RunGceAssertionCredentials(service_account_name=service_account_name, cache_filename=tempname, scopes=scopes)
            pre_cache_call_count = opener_mock.call_count
            creds2 = self._RunGceAssertionCredentials(service_account_name=service_account_name, cache_filename=tempname, scopes=None)
        finally:
            shutil.rmtree(tempd)
    self.assertEqual(creds1.client_id, creds2.client_id)
    self.assertEqual(pre_cache_call_count, 3)
    self.assertEqual(opener_mock.call_count, 4)