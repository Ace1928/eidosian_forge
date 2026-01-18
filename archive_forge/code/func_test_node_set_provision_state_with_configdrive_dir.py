import copy
import tempfile
import time
from unittest import mock
import testtools
from testtools.matchers import HasLength
from ironicclient.common import utils as common_utils
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import node
from ironicclient.v1 import volume_connector
from ironicclient.v1 import volume_target
@mock.patch.object(common_utils, 'make_configdrive', autospec=True)
def test_node_set_provision_state_with_configdrive_dir(self, mock_configdrive):
    mock_configdrive.return_value = 'fake-configdrive'
    target_state = 'active'
    with common_utils.tempdir() as dirname:
        self.mgr.set_provision_state(NODE1['uuid'], target_state, configdrive=dirname)
        mock_configdrive.assert_called_once_with(dirname)
    body = {'target': target_state, 'configdrive': 'fake-configdrive'}
    expect = [('PUT', '/v1/nodes/%s/states/provision' % NODE1['uuid'], {}, body)]
    self.assertEqual(expect, self.api.calls)