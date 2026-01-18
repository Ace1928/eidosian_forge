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
def test_list_schema_int_good(self):
    list_schema = {'Type': 'Integer'}
    p = properties.Property({'Type': 'List', 'Schema': list_schema})
    self.assertEqual([1, 2, 3], p.get_value([1, 2, 3]))