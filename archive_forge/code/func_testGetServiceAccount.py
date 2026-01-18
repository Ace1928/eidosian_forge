import json
import os.path
import shutil
import tempfile
import unittest
import mock
import six
from apitools.base.py import credentials_lib
from apitools.base.py import util
def testGetServiceAccount(self):
    creds = self._GetServiceCreds()
    opener = mock.MagicMock()
    opener.open = mock.MagicMock()
    opener.open.return_value = six.StringIO('default/\nanother')
    with mock.patch.object(six.moves.urllib.request, 'build_opener', return_value=opener, autospec=True) as build_opener:
        creds.GetServiceAccount('default')
        self.assertEqual(1, build_opener.call_count)
        self.assertEqual(1, opener.open.call_count)
        req = opener.open.call_args[0][0]
        self.assertTrue(req.get_full_url().startswith('http://metadata.google.internal/'))
        self.assertEqual('Google', req.get_header('Metadata-flavor'))