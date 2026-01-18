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
def test_from_string_param(self):
    description = 'WebServer EC2 instance type'
    allowed_values = ['t1.micro', 'm1.small', 'm1.large', 'm1.xlarge', 'm2.xlarge', 'm2.2xlarge', 'm2.4xlarge', 'c1.medium', 'c1.xlarge', 'cc1.4xlarge']
    constraint_desc = 'Must be a valid EC2 instance type.'
    param = parameters.Schema.from_dict('name', {'Type': 'String', 'Description': description, 'Default': 'm1.large', 'AllowedValues': allowed_values, 'ConstraintDescription': constraint_desc})
    schema = properties.Schema.from_parameter(param)
    self.assertEqual(properties.Schema.STRING, schema.type)
    self.assertEqual(description, schema.description)
    self.assertIsNone(schema.default)
    self.assertFalse(schema.required)
    self.assertEqual(1, len(schema.constraints))
    allowed_constraint = schema.constraints[0]
    self.assertEqual(tuple(allowed_values), allowed_constraint.allowed)
    self.assertEqual(constraint_desc, allowed_constraint.description)
    props = properties.Properties({'test': schema}, {})
    props.validate()