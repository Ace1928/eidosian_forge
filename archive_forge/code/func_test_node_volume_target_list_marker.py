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
def test_node_volume_target_list_marker(self):
    self.api = utils.FakeAPI(fake_responses_pagination)
    self.mgr = node.NodeManager(self.api)
    volume_targets = self.mgr.list_volume_targets(NODE1['uuid'], marker=TARGET['uuid'])
    expect = [('GET', '/v1/nodes/%s/volume/targets?marker=%s' % (NODE1['uuid'], TARGET['uuid']), {}, None)]
    self._validate_node_volume_target_list(expect, volume_targets)