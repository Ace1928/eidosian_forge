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
def test_get_res_condition_invalid(self):
    tmpl = copy.deepcopy(self.tmpl)
    stk = stack.Stack(self.ctx, 'test_res_invalid_condition', tmpl)
    conds = tmpl.conditions(stk)
    ex = self.assertRaises(ValueError, conds.is_enabled, 'invalid_cd')
    self.assertIn('Invalid condition "invalid_cd"', str(ex))
    ex = self.assertRaises(ValueError, conds.is_enabled, 111)
    self.assertIn('Invalid condition "111"', str(ex))