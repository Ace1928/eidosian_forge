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
def test_node_set_provision_state_with_rescue_password(self):
    rescue_password = 'supersecret'
    target_state = 'rescue'
    self.mgr.set_provision_state(NODE1['uuid'], target_state, rescue_password=rescue_password)
    body = {'target': target_state, 'rescue_password': rescue_password}
    expect = [('PUT', '/v1/nodes/%s/states/provision' % NODE1['uuid'], {}, body)]
    self.assertEqual(expect, self.api.calls)