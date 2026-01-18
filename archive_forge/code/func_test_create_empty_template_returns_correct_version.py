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
def test_create_empty_template_returns_correct_version(self):
    t = template_format.parse('\n            AWSTemplateFormatVersion: 2010-09-09\n            Parameters:\n            Resources:\n            Outputs:\n            ')
    aws_tmpl = template.Template(t)
    empty_template = template.Template.create_empty_template(version=aws_tmpl.version)
    self.assertEqual(aws_tmpl.__class__, empty_template.__class__)
    self.assertEqual({}, empty_template['Mappings'])
    self.assertEqual({}, empty_template['Resources'])
    self.assertEqual({}, empty_template['Outputs'])
    t = template_format.parse('\n            HeatTemplateFormatVersion: 2012-12-12\n            Parameters:\n            Resources:\n            Outputs:\n            ')
    heat_tmpl = template.Template(t)
    empty_template = template.Template.create_empty_template(version=heat_tmpl.version)
    self.assertEqual(heat_tmpl.__class__, empty_template.__class__)
    self.assertEqual({}, empty_template['Mappings'])
    self.assertEqual({}, empty_template['Resources'])
    self.assertEqual({}, empty_template['Outputs'])
    t = template_format.parse('\n            heat_template_version: 2015-04-30\n            parameter_groups:\n            resources:\n            outputs:\n            ')
    hot_tmpl = template.Template(t)
    empty_template = template.Template.create_empty_template(version=hot_tmpl.version)
    self.assertEqual(hot_tmpl.__class__, empty_template.__class__)
    self.assertEqual({}, empty_template['parameter_groups'])
    self.assertEqual({}, empty_template['resources'])
    self.assertEqual({}, empty_template['outputs'])