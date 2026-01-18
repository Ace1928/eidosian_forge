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
def test_custom_index_var(self):
    templ = copy.deepcopy(template_repl)
    templ['resources']['group1']['properties']['index_var'] = '__foo__'
    stack = utils.parse_stack(templ)
    snip = stack.t.resource_definitions(stack)['group1']
    resg = resource_group.ResourceGroup('test', snip, stack)
    expect = {'heat_template_version': '2015-04-30', 'resources': {'0': {'type': 'ResourceWithListProp%index%', 'properties': {'Foo': 'Bar_%index%', 'listprop': ['%index%_0', '%index%_1', '%index%_2']}}}, 'outputs': {'refs_map': {'value': {'0': {'get_resource': '0'}}}}}
    nested = resg._assemble_nested(['0']).t
    res_prop = nested['resources']['0']['properties']
    res_prop['listprop'] = list(res_prop['listprop'])
    self.assertEqual(expect, nested)
    props = copy.deepcopy(templ['resources']['group1']['properties'])
    res_def = props['resource_def']
    res_def['properties']['Foo'] = 'Bar___foo__'
    res_def['properties']['listprop'] = ['__foo___0', '__foo___1', '__foo___2']
    res_def['type'] = 'ResourceWithListProp__foo__'
    snip = snip.freeze(properties=props)
    resg = resource_group.ResourceGroup('test', snip, stack)
    expect = {'heat_template_version': '2015-04-30', 'resources': {'0': {'type': 'ResourceWithListProp__foo__', 'properties': {'Foo': 'Bar_0', 'listprop': ['0_0', '0_1', '0_2']}}}, 'outputs': {'refs_map': {'value': {'0': {'get_resource': '0'}}}}}
    nested = resg._assemble_nested(['0']).t
    res_prop = nested['resources']['0']['properties']
    res_prop['listprop'] = list(res_prop['listprop'])
    self.assertEqual(expect, nested)