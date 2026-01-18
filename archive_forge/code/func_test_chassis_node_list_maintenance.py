import copy
import testtools
from testtools.matchers import HasLength
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.chassis
def test_chassis_node_list_maintenance(self):
    nodes = self.mgr.list_nodes(CHASSIS['uuid'], maintenance=False)
    expect = [('GET', '/v1/chassis/%s/nodes?maintenance=False' % CHASSIS['uuid'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(1, len(nodes))