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
def test_from_json_param(self):
    param = parameters.Schema.from_dict('name', {'Type': 'Json', 'Default': {'foo': 'bar', 'blarg': 'wibble'}})
    schema = properties.Schema.from_parameter(param)
    self.assertEqual(properties.Schema.MAP, schema.type)
    self.assertIsNone(schema.default)
    self.assertFalse(schema.required)
    self.assertTrue(schema.allow_conversion)
    props = properties.Properties({'test': schema}, {})
    props.validate()