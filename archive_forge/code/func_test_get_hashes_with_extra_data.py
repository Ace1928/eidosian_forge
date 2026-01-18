import collections
import copy
import datetime
import hashlib
import inspect
from unittest import mock
import iso8601
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
def test_get_hashes_with_extra_data(self):
    fp = 'garyoak'
    mock_func = mock.MagicMock()
    with mock.patch.object(self.ovc, '_get_fingerprint') as mock_gf:
        mock_gf.return_value = fp
        actual = self.ovc.get_hashes(extra_data_func=mock_func)
    expected = self._generate_hashes(self.obj_classes, fp)
    expected_calls = [((name,), {'extra_data_func': mock_func}) for name in self.obj_classes.keys()]
    self.assertEqual(expected, actual, 'ObjectVersionChecker is not getting the fingerprints of all registered objects.')
    self.assertEqual(len(expected_calls), len(mock_gf.call_args_list), 'get_hashes() did not call get the fingerprints of all objects in the registry.')
    for call in expected_calls:
        self.assertIn(call, mock_gf.call_args_list, 'get_hashes() did not call _get_fingerprint()correctly.')