import sys
from unittest import mock
from stevedore import _cache
from stevedore.tests import utils
@mock.patch('os.makedirs')
@mock.patch('builtins.open')
def test__get_data_for_path_no_write(self, mock_open, mock_mkdir):
    sot = _cache.Cache()
    sot._disable_caching = True
    mock_open.side_effect = IOError
    sot._get_data_for_path('fake')
    mock_mkdir.assert_not_called()