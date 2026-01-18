import copy
import hashlib
import json
import fixtures
from stevedore import extension
from heat.common import exception
from heat.common import template_format
from heat.engine.cfn import functions as cfn_funcs
from heat.engine.cfn import parameters as cfn_p
from heat.engine.cfn import template as cfn_t
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import function
from heat.engine.hot import template as hot_t
from heat.engine import node_data
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import stk_defn
from heat.engine import template
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_conditions_circular_ref(self):
    t = {'heat_template_version': '2016-10-14', 'parameters': {'env_type': {'type': 'string', 'default': 'test'}}, 'conditions': {'first_cond': {'not': 'second_cond'}, 'second_cond': {'not': 'third_cond'}, 'third_cond': {'not': 'first_cond'}}}
    tmpl = template.Template(t)
    stk = stack.Stack(self.ctx, 'test_condition_circular_ref', tmpl)
    conds = tmpl.conditions(stk)
    ex = self.assertRaises(exception.StackValidationFailed, conds.is_enabled, 'first_cond')
    self.assertIn('Circular definition for condition "first_cond"', str(ex))