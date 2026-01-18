import os
from unittest import mock
import glance_store
from oslo_config import cfg
from oslo_utils.fixture import uuidsentinel as uuids
from glance.common import exception
from glance import context
from glance import housekeeping
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
@mock.patch('os.remove')
@mock.patch.object(housekeeping, 'LOG')
def test_delete_file_failed(self, mock_LOG, mock_remove):
    os.remove.side_effect = Exception('insufficient plutonium')
    self.assertFalse(self.cleaner.delete_file('foo'))
    os.remove.assert_called_once_with('foo')
    mock_LOG.error.assert_called_once_with('Failed to delete stale staging path %(path)r: %(err)s', {'path': 'foo', 'err': 'insufficient plutonium'})