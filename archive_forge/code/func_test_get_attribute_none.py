from unittest import mock
from heat.engine import attributes
from heat.engine import resources
from heat.engine import support
from heat.tests import common
def test_get_attribute_none(self):
    """Test that we get the attribute values we expect."""
    self.resolver.return_value = None
    attribs = attributes.Attributes('test resource', self.attributes_schema, self.resolver)
    self.assertIsNone(attribs['test1'])
    self.resolver.assert_called_once_with('test1')