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
def test_node_set_provision_state_with_cleansteps(self):
    cleansteps = [{'step': 'upgrade', 'interface': 'deploy'}]
    target_state = 'clean'
    self.mgr.set_provision_state(NODE1['uuid'], target_state, cleansteps=cleansteps)
    body = {'target': target_state, 'clean_steps': cleansteps}
    expect = [('PUT', '/v1/nodes/%s/states/provision' % NODE1['uuid'], {}, body)]
    self.assertEqual(expect, self.api.calls)