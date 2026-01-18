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
def test_create_empty_template_from_another_template(self):
    res_param_template = template_format.parse('{\n          "HeatTemplateFormatVersion" : "2012-12-12",\n          "Parameters" : {\n            "foo" : { "Type" : "String" },\n            "blarg" : { "Type" : "String", "Default": "quux" }\n          },\n          "Resources" : {\n            "foo" : { "Type" : "GenericResourceType" },\n            "blarg" : { "Type" : "GenericResourceType" }\n          }\n        }')
    env = environment.Environment({'foo': 'bar'})
    hot_tmpl = template.Template(res_param_template, env)
    empty_template = template.Template.create_empty_template(from_template=hot_tmpl)
    self.assertEqual({}, empty_template['Resources'])
    self.assertEqual(hot_tmpl.env, empty_template.env)