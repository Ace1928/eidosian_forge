from heat.common import exception
from heat.common import template_format
from heat.engine.cfn import functions as cfn_funcs
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine import properties
from heat.engine import rsrc_defn
from heat.tests import common
from heat.tests import utils
def get_param_defn(value):
    stack = FakeStack({'Foo': value})
    param_func = hot_funcs.GetParam(stack, 'get_param', 'Foo')
    return rsrc_defn.ResourceDefinition('rsrc', 'SomeType', {'Foo': param_func})