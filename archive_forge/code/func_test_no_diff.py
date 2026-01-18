from heat.common import exception
from heat.common import template_format
from heat.engine.cfn import functions as cfn_funcs
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine import properties
from heat.engine import rsrc_defn
from heat.tests import common
from heat.tests import utils
def test_no_diff(self):
    before = rsrc_defn.ResourceDefinition('rsrc', 'SomeType', properties={'Foo': 'blarg'}, update_policy={'bar': 'quux'}, metadata={'baz': 'wibble'}, depends=['other_resource'], deletion_policy='Delete')
    after = rsrc_defn.ResourceDefinition('rsrc', 'SomeType', properties={'Foo': 'blarg'}, update_policy={'bar': 'quux'}, metadata={'baz': 'wibble'}, depends=['other_other_resource'], deletion_policy='Retain')
    diff = after - before
    self.assertFalse(diff.properties_changed())
    self.assertFalse(diff.update_policy_changed())
    self.assertFalse(diff.metadata_changed())
    self.assertFalse(diff)