from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.openstack.heat import delay
from heat.engine import status
from heat.tests import common
from heat.tests import utils
from oslo_utils import fixture as utils_fixture
from oslo_utils import timeutils
def test_delay_params(self):
    stk = utils.parse_stack(self.simple_template)
    self.assertEqual((3, 0), stk['constant']._delay_parameters())
    self.assertEqual((1.6, 4.2), stk['variable']._delay_parameters())
    min_wait, max_jitter = stk['variable_prod']._delay_parameters()
    self.assertEqual(2, min_wait)
    self.assertAlmostEqual(66.6, max_jitter)