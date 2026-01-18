from neutron_lib.agent.common import constants
from neutron_lib.agent.common import utils
from neutron_lib.tests import _base
def test_no_registers_defined(self):
    flow = {'foo': 'bar'}
    utils.create_reg_numbers(flow)
    self.assertEqual({'foo': 'bar'}, flow)