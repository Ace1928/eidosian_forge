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
def test_node_set_boot_mode_bios(self):
    target_state = 'bios'
    self.mgr.set_boot_mode(NODE1['uuid'], target_state)
    body = {'target': target_state}
    expect = [('PUT', '/v1/nodes/%s/states/boot_mode' % NODE1['uuid'], {}, body)]
    self.assertEqual(expect, self.api.calls)