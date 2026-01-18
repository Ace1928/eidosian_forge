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
def test_from_number_param_allowed_vals(self):
    constraint_desc = 'The quick brown fox jumps over the lazy dog.'
    param = parameters.Schema.from_dict('name', {'Type': 'Number', 'Default': '42', 'AllowedValues': ['10', '42', '100'], 'ConstraintDescription': constraint_desc})
    schema = properties.Schema.from_parameter(param)
    self.assertEqual(properties.Schema.NUMBER, schema.type)
    self.assertIsNone(schema.default)
    self.assertFalse(schema.required)
    self.assertEqual(1, len(schema.constraints))
    self.assertFalse(schema.allow_conversion)
    allowed_constraint = schema.constraints[0]
    self.assertEqual(('10', '42', '100'), allowed_constraint.allowed)
    self.assertEqual(constraint_desc, allowed_constraint.description)
    props = properties.Properties({'test': schema}, {})
    props.validate()