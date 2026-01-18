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
def test_compare_different(self):
    schema = {'foo': {'Type': 'Integer'}}
    props_a = properties.Properties(schema, {'foo': 0})
    props_b = properties.Properties(schema, {'foo': 1})
    self.assertTrue(props_a != props_b)