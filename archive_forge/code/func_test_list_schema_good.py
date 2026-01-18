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
def test_list_schema_good(self):
    map_schema = {'valid': {'Type': 'Boolean'}}
    list_schema = {'Type': 'Map', 'Schema': map_schema}
    p = properties.Property({'Type': 'List', 'Schema': list_schema})
    self.assertEqual([{'valid': True}, {'valid': False}], p.get_value([{'valid': 'TRUE'}, {'valid': 'False'}]))