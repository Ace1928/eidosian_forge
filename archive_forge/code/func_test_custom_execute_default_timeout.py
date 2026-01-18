from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick import privileged
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base
@mock.patch('threading.Timer')
def test_custom_execute_default_timeout(self, mock_timer):
    """Confirm timeout defaults to 600 and the thread timer is started."""
    priv_rootwrap.custom_execute('echo', 'hola')
    mock_timer.assert_called_once_with(600, mock.ANY, mock.ANY)
    mock_timer.return_value.start.assert_called_once_with()