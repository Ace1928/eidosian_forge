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
def test_invalid_removal_policies_nolist(self):
    """Test that error raised for malformed removal_policies."""
    tmp = copy.deepcopy(template)
    grp_props = tmp['resources']['group1']['properties']
    grp_props['removal_policies'] = 'notallowed'
    stack = utils.parse_stack(tmp)
    snip = stack.t.resource_definitions(stack)['group1']
    resg = resource_group.ResourceGroup('test', snip, stack)
    exc = self.assertRaises(exception.StackValidationFailed, resg.validate)
    errstr = 'removal_policies: "\'notallowed\'" is not a list'
    self.assertIn(errstr, str(exc))