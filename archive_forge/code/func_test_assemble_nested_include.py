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
def test_assemble_nested_include(self):
    templ = copy.deepcopy(template)
    res_def = templ['resources']['group1']['properties']['resource_def']
    res_def['properties']['Foo'] = None
    stack = utils.parse_stack(templ)
    snip = stack.t.resource_definitions(stack)['group1']
    resg = resource_group.ResourceGroup('test', snip, stack)
    expect = {'heat_template_version': '2015-04-30', 'resources': {'0': {'type': 'OverwrittenFnGetRefIdType', 'properties': {}}}, 'outputs': {'refs_map': {'value': {'0': {'get_resource': '0'}}}}}
    self.assertEqual(expect, resg._assemble_nested(['0']).t)
    expect['resources']['0']['properties'] = {'Foo': None}
    self.assertEqual(expect, resg._assemble_nested(['0'], include_all=True).t)