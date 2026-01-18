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
def test_condition_reference_condition(self):
    t = {'heat_template_version': '2016-10-14', 'parameters': {'env_type': {'type': 'string', 'default': 'test'}}, 'conditions': {'prod_env': {'equals': [{'get_param': 'env_type'}, 'prod']}, 'test_env': {'not': 'prod_env'}, 'prod_or_test_env': {'or': ['prod_env', 'test_env']}, 'prod_and_test_env': {'and': ['prod_env', 'test_env']}}}
    tmpl = template.Template(t)
    stk = stack.Stack(self.ctx, 'test_condition_reference', tmpl)
    conditions = tmpl.conditions(stk)
    self.assertFalse(conditions.is_enabled('prod_env'))
    self.assertTrue(conditions.is_enabled('test_env'))
    self.assertTrue(conditions.is_enabled('prod_or_test_env'))
    self.assertFalse(conditions.is_enabled('prod_and_test_env'))