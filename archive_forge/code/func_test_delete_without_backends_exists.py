import sys
from unittest import mock
import urllib.error
from glance_store import exceptions as store_exceptions
from oslo_config import cfg
from oslo_utils import units
import taskflow
import glance.async_.flows.api_image_import as import_flow
from glance.common import exception
from glance.common.scripts.image_import import main as image_import
from glance import context
from glance.domain import ExtraProperties
from glance import gateway
import glance.tests.utils as test_utils
from cursive import exception as cursive_exception
@mock.patch('os.path.exists')
@mock.patch('os.unlink')
def test_delete_without_backends_exists(self, mock_unlink, mock_exists):
    mock_exists.return_value = True
    task = import_flow._DeleteFromFS(TASK_ID1, TASK_TYPE)
    task.execute('1234567foo')
    mock_unlink.assert_called_once_with('foo')
    mock_unlink.reset_mock()
    mock_unlink.side_effect = OSError(123, 'failed')
    task.execute('1234567foo')