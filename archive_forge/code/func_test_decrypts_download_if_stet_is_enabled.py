from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import stat
from gslib import storage_url
from gslib.tests import testcase
from gslib.tests import util
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import unittest
from gslib.utils import system_util
from gslib.utils import temporary_file_util
def test_decrypts_download_if_stet_is_enabled(self):
    object_uri = self.CreateObject(contents='abc')
    test_file = self.CreateTempFile()
    stderr = self.RunGsUtil(['-o', 'GSUtil:stet_binary_path={}'.format(self.stet_binary_path), '-o', 'GSUtil:stet_config_path={}'.format(self.stet_config_path), 'cp', '--stet', suri(object_uri), test_file], return_stderr=True)
    self.assertNotIn('/4.0 B]', stderr)
    with open(test_file) as file_reader:
        downloaded_text = file_reader.read()
    self.assertIn('subcommand: decrypt', downloaded_text)
    self.assertIn('config file: --config-file={}'.format(self.stet_config_path), downloaded_text)
    self.assertIn('blob id: --blob-id={}'.format(suri(object_uri)), downloaded_text)
    self.assertIn('in file: {}'.format(test_file), downloaded_text)
    self.assertIn('out file: {}_.stet_tmp'.format(test_file), downloaded_text)
    self.assertFalse(os.path.exists(temporary_file_util.GetStetTempFileName(storage_url.StorageUrlFromString(test_file))))