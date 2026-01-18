from unittest import mock
from oslo_serialization import jsonutils
from heat.common import exception
from heat.engine import constraints
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine.hot import parameters as hot_param
from heat.engine import parameters
from heat.engine import plugin_manager
from heat.engine import properties
from heat.engine import resources
from heat.engine import rsrc_defn
from heat.engine import support
from heat.engine import translation
from heat.tests import common
def test_nested_properties_schema_invalid_property_in_list(self):
    child_schema = {'Key': {'Type': 'String', 'Required': True}, 'Value': {'Type': 'Boolean', 'Default': True}}
    list_schema = {'Type': 'Map', 'Schema': child_schema}
    schema = {'foo': {'Type': 'List', 'Schema': list_schema}}
    valid_data = {'foo': [{'Key': 'Test'}]}
    props = properties.Properties(schema, valid_data)
    self.assertIsNone(props.validate())
    invalid_data = {'foo': [{'Key': 'Test', 'bar': 'baz'}]}
    props = properties.Properties(schema, invalid_data)
    ex = self.assertRaises(exception.StackValidationFailed, props.validate)
    self.assertEqual('Property error: foo[0]: Unknown Property bar', str(ex))