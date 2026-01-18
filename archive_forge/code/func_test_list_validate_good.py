from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_list_validate_good(self):
    schema = {'Type': 'CommaDelimitedList'}
    val = ['foo', 'bar', 'baz']
    val_s = 'foo,bar,baz'
    p = new_parameter('p', schema, val_s, validate_value=False)
    p.validate()
    self.assertEqual(val, p.value())
    self.assertEqual(val, p.parsed)