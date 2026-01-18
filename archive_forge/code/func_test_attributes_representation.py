from unittest import mock
from heat.engine import attributes
from heat.engine import resources
from heat.engine import support
from heat.tests import common
def test_attributes_representation(self):
    """Test that attributes are displayed correct."""
    self.resolver.return_value = 'value1'
    attribs = attributes.Attributes('test resource', self.attributes_schema, self.resolver)
    msg = 'Attributes for test resource:\n\tvalue1\n\tvalue1\n\tvalue1'
    self.assertEqual(msg, str(attribs))
    calls = [mock.call('test1'), mock.call('test2'), mock.call('test3')]
    self.resolver.assert_has_calls(calls, any_order=True)