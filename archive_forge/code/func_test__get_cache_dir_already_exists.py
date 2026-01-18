import os
from unittest import mock
import dogpile.cache
from ironicclient.common import filecache
from ironicclient.tests.unit import utils
@mock.patch.object(filecache, 'CACHE', 5552368)
@mock.patch.object(os.path, 'exists', autospec=True)
@mock.patch.object(os, 'makedirs', autospec=True)
def test__get_cache_dir_already_exists(self, mock_makedirs, mock_exists):
    mock_exists.return_value = True
    self.assertEqual(5552368, filecache._get_cache())
    self.assertEqual(5552368, filecache.CACHE)
    self.assertEqual(0, mock_exists.call_count)
    self.assertEqual(0, mock_makedirs.call_count)