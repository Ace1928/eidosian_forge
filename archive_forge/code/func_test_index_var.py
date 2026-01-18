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
def test_index_var(self):
    stack = utils.parse_stack(template_repl)
    snip = stack.t.resource_definitions(stack)['group1']
    resg = resource_group.ResourceGroup('test', snip, stack)
    expect = {'heat_template_version': '2015-04-30', 'resources': {'0': {'type': 'ResourceWithListProp%index%', 'properties': {'Foo': 'Bar_0', 'listprop': ['0_0', '0_1', '0_2']}}, '1': {'type': 'ResourceWithListProp%index%', 'properties': {'Foo': 'Bar_1', 'listprop': ['1_0', '1_1', '1_2']}}, '2': {'type': 'ResourceWithListProp%index%', 'properties': {'Foo': 'Bar_2', 'listprop': ['2_0', '2_1', '2_2']}}}, 'outputs': {'refs_map': {'value': {'0': {'get_resource': '0'}, '1': {'get_resource': '1'}, '2': {'get_resource': '2'}}}}}
    nested = resg._assemble_nested(['0', '1', '2']).t
    for res in nested['resources']:
        res_prop = nested['resources'][res]['properties']
        res_prop['listprop'] = list(res_prop['listprop'])
    self.assertEqual(expect, nested)