import copy
from unittest import mock
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_group
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_assemble_nested_rolling_update_none(self):
    expect = {'heat_template_version': '2015-04-30', 'resources': {'0': {'type': 'OverwrittenFnGetRefIdType', 'properties': {'foo': 'bar'}}, '1': {'type': 'OverwrittenFnGetRefIdType', 'properties': {'foo': 'bar'}}}, 'outputs': {'refs_map': {'value': {'0': {'get_resource': '0'}, '1': {'get_resource': '1'}}}}}
    resource_def = rsrc_defn.ResourceDefinition(None, 'OverwrittenFnGetRefIdType', {'foo': 'baz'})
    stack = utils.parse_stack(template)
    snip = stack.t.resource_definitions(stack)['group1']
    resg = resource_group.ResourceGroup('test', snip, stack)
    nested = get_fake_nested_stack(['0', '1'])
    self.inspector.template.return_value = nested.defn._template
    self.inspector.member_names.return_value = ['0', '1']
    resg.build_resource_definition = mock.Mock(return_value=resource_def)
    self.assertEqual(expect, resg._assemble_for_rolling_update(2, 0).t)