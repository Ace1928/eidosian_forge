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
def test_add_resource(self):
    cfn_tpl = template_format.parse('\n        AWSTemplateFormatVersion: 2010-09-09\n        Resources:\n          resource1:\n            Type: AWS::EC2::Instance\n            Properties:\n              property1: value1\n            Metadata:\n              foo: bar\n            DependsOn: dummy\n            DeletionPolicy: Retain\n            UpdatePolicy:\n              foo: bar\n          resource2:\n            Type: AWS::EC2::Instance\n          resource3:\n            Type: AWS::EC2::Instance\n            DependsOn:\n              - resource1\n              - dummy\n              - resource2\n        ')
    source = template.Template(cfn_tpl)
    empty = template.Template(copy.deepcopy(empty_template))
    stk = stack.Stack(self.ctx, 'test_stack', source)
    for rname, defn in sorted(source.resource_definitions(stk).items()):
        empty.add_resource(defn)
    expected = copy.deepcopy(cfn_tpl['Resources'])
    del expected['resource1']['DependsOn']
    expected['resource3']['DependsOn'] = ['resource1', 'resource2']
    self.assertEqual(expected, empty.t['Resources'])