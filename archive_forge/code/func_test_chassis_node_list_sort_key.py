import copy
import testtools
from testtools.matchers import HasLength
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.chassis
def test_chassis_node_list_sort_key(self):
    self.api = utils.FakeAPI(fake_responses_sorting)
    self.mgr = ironicclient.v1.chassis.ChassisManager(self.api)
    nodes = self.mgr.list_nodes(CHASSIS['uuid'], sort_key='updated_at')
    expect = [('GET', '/v1/chassis/%s/nodes?sort_key=updated_at' % CHASSIS['uuid'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertThat(nodes, HasLength(1))
    self.assertEqual(NODE['uuid'], nodes[0].uuid)