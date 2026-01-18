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
def test_from_param_string_min_max_len(self):
    param = parameters.Schema.from_dict('name', {'Description': 'WebServer EC2 instance type', 'Type': 'String', 'Default': 'm1.large', 'MinLength': '7', 'MaxLength': '11'})
    schema = properties.Schema.from_parameter(param)
    self.assertFalse(schema.required)
    self.assertIsNone(schema.default)
    self.assertEqual(1, len(schema.constraints))
    len_constraint = schema.constraints[0]
    self.assertEqual(7, len_constraint.min)
    self.assertEqual(11, len_constraint.max)
    props = properties.Properties({'test': schema}, {})
    props.validate()