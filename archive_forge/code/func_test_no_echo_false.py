from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_no_echo_false(self):
    p = new_parameter('echoic', {'Type': self.p_type, 'NoEcho': 'false'}, self.value)
    self.assertFalse(p.hidden())
    if self.p_type == 'Json':
        self.assertEqual(json.loads(self.expected), json.loads(str(p)))
    else:
        self.assertEqual(self.expected, str(p))