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
def test_node_list_sort_dir(self):
    self.api = utils.FakeAPI(fake_responses_sorting)
    self.mgr = node.NodeManager(self.api)
    nodes = self.mgr.list(sort_dir='desc')
    expect = [('GET', '/v1/nodes/?sort_dir=desc', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(2, len(nodes))