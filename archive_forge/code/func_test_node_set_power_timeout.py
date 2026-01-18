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
def test_node_set_power_timeout(self):
    power_state = self.mgr.set_power_state(NODE1['uuid'], 'off', timeout=2)
    body = {'target': 'power off', 'timeout': 2}
    expect = [('PUT', '/v1/nodes/%s/states/power' % NODE1['uuid'], {}, body)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual('power off', power_state.target_power_state)