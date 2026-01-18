from unittest import mock
from heat.engine import attributes
from heat.engine import resources
from heat.engine import support
from heat.tests import common
def test_get_attribute_nonexist(self):
    """Test that we get the attribute values we expect."""
    self.resolver.return_value = 'value1'
    attribs = attributes.Attributes('test resource', self.attributes_schema, self.resolver)
    self.assertRaises(KeyError, attribs.__getitem__, 'not there')
    self.assertFalse(self.resolver.called)