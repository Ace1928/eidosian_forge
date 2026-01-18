import platform
import sys
from unittest import mock
from oslo_concurrency import processutils as putils
from oslo_service import loopingcall
from os_brick import exception
from os_brick.initiator import connector
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import fake
from os_brick.initiator.connectors import iscsi
from os_brick.initiator.connectors import nvmeof
from os_brick.initiator import linuxfc
from os_brick.privileged import nvmeof as priv_nvmeof
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base as test_base
from os_brick import utils
@mock.patch.object(priv_rootwrap, 'custom_execute', side_effect=OSError(2))
@mock.patch.object(priv_rootwrap, 'execute', return_value=('', ''))
def test_brick_get_connector_properties_multipath(self, mock_execute, mock_custom_execute):
    self._test_brick_get_connector_properties(True, True, True)
    mock_execute.assert_called_once_with('multipathd', 'show', 'status', run_as_root=True, root_helper='sudo')
    mock_custom_execute.assert_called_once_with('nvme', 'version')