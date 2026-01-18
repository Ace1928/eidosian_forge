from unittest import mock
from heat.engine import attributes
from heat.engine import resources
from heat.engine import support
from heat.tests import common
def test_as_outputs_hot(self):
    """Test that Output format works as expected."""
    expected = {'test1': {'value': {'get_attr': ['test_resource', 'test1']}, 'description': 'Test attrib 1'}, 'test2': {'value': {'get_attr': ['test_resource', 'test2']}, 'description': 'Test attrib 2'}, 'test3': {'value': {'get_attr': ['test_resource', 'test3']}, 'description': 'Test attrib 3'}, 'OS::stack_id': {'value': {'get_resource': 'test_resource'}}}
    MyTestResourceClass = mock.MagicMock()
    MyTestResourceClass.attributes_schema = {'test1': attributes.Schema('Test attrib 1'), 'test2': attributes.Schema('Test attrib 2'), 'test3': attributes.Schema('Test attrib 3'), 'test4': attributes.Schema('Test attrib 4', support_status=support.SupportStatus(status=support.HIDDEN))}
    self.assertEqual(expected, attributes.Attributes.as_outputs('test_resource', MyTestResourceClass, 'hot'))