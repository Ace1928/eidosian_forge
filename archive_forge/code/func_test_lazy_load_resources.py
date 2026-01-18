from unittest import mock
from oslo_config import cfg
from oslo_messaging.rpc import dispatcher
from heat.common import exception
from heat.common import identifier
from heat.engine.clients.os import heat_plugin
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import dependencies
from heat.engine import resource as res
from heat.engine.resources.aws.ec2 import instance as ins
from heat.engine import service
from heat.engine import stack
from heat.engine import stack_lock
from heat.engine import template as templatem
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_lazy_load_resources(self):
    stack_name = 'lazy_load_test'
    lazy_load_template = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'foo': {'Type': 'GenericResourceType'}, 'bar': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': {'Ref': 'foo'}}}}}
    templ = templatem.Template(lazy_load_template)
    stk = stack.Stack(self.ctx, stack_name, templ)
    self.assertIsNone(stk._resources)
    self.assertIsNone(stk._dependencies)
    resources = stk.resources
    self.assertIsInstance(resources, dict)
    self.assertEqual(2, len(resources))
    self.assertIsInstance(resources.get('foo'), generic_rsrc.GenericResource)
    self.assertIsInstance(resources.get('bar'), generic_rsrc.ResourceWithProps)
    stack_dependencies = stk.dependencies
    self.assertIsInstance(stack_dependencies, dependencies.Dependencies)
    self.assertEqual(2, len(stack_dependencies.graph()))