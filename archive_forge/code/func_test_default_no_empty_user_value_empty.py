from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_default_no_empty_user_value_empty(self):
    p = new_parameter('defaulted', {'Type': self.p_type, 'Default': self.default}, self.zero)
    self.assertTrue(p.has_default())
    self.assertEqual(self.default, p.default())
    self.assertEqual(self.zero, p.value())