from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_list_value_list_good(self):
    schema = {'Type': 'CommaDelimitedList', 'AllowedValues': ['foo', 'bar', 'baz']}
    p = new_parameter('p', schema, 'baz,foo,bar')
    self.assertEqual('baz,foo,bar'.split(','), p.value())
    schema['Default'] = []
    p = new_parameter('p', schema)
    self.assertEqual([], p.value())
    schema['Default'] = 'baz,foo,bar'
    p = new_parameter('p', schema)
    self.assertEqual('baz,foo,bar'.split(','), p.value())
    schema['AllowedValues'] = ['1', '3', '5']
    schema['Default'] = []
    p = new_parameter('p', schema, [1, 3, 5])
    self.assertEqual('1,3,5', str(p))
    schema['Default'] = [1, 3, 5]
    p = new_parameter('p', schema)
    self.assertEqual('1,3,5'.split(','), p.value())