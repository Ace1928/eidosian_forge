from heat.common import exception
from heat.common import template_format
from heat.engine.cfn import functions as cfn_funcs
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine import properties
from heat.engine import rsrc_defn
from heat.tests import common
from heat.tests import utils
def test_hash_types(self):
    rd1 = rsrc_defn.ResourceDefinition('rsrc', 'SomeType1')
    rd2 = rsrc_defn.ResourceDefinition('rsrc', 'SomeType2')
    self.assertNotEqual(rd1, rd2)
    self.assertNotEqual(hash(rd1), hash(rd2))