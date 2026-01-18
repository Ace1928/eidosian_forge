from heat.common import exception
from heat.common import template_format
from heat.engine.cfn import functions as cfn_funcs
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine import properties
from heat.engine import rsrc_defn
from heat.tests import common
from heat.tests import utils
def test_dependencies_default(self):
    rd = rsrc_defn.ResourceDefinition('rsrc', 'SomeType')
    stack = {'foo': 'FOO', 'bar': 'BAR'}
    self.assertEqual(set(), rd.required_resource_names())
    self.assertEqual([], list(rd.dependencies(stack)))