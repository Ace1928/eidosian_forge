import copy
import testtools
from testtools.matchers import HasLength
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.chassis
def test_chassis_node_list_associated(self):
    nodes = self.mgr.list_nodes(CHASSIS['uuid'], associated=True)
    expect = [('GET', '/v1/chassis/%s/nodes?associated=True' % CHASSIS['uuid'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(1, len(nodes))