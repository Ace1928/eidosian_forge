from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_list_value_list_default_empty(self):
    schema = {'Type': 'CommaDelimitedList', 'Default': ''}
    p = new_parameter('p', schema)
    self.assertEqual([], p.value())