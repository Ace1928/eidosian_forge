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
def test_node_list_conductor_group(self):
    nodes = self.mgr.list(conductor_group='foo')
    expect = [('GET', '/v1/nodes/?conductor_group=foo', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertThat(nodes, HasLength(1))
    self.assertEqual(NODE1['uuid'], getattr(nodes[0], 'uuid'))