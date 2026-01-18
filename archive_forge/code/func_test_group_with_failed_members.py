from unittest import mock
from heat.common import grouputils
from heat.common import identifier
from heat.common import template_format
from heat.rpc import client as rpc_client
from heat.tests import common
from heat.tests import utils
def test_group_with_failed_members(self):
    group = mock.Mock()
    t = template_format.parse(nested_stack)
    stack = utils.parse_stack(t)
    self.patchobject(group, 'nested', return_value=stack)
    rsrc_err = stack.resources['r0']
    rsrc_err.status = rsrc_err.FAILED
    rsrc_ok = stack.resources['r1']
    self.assertEqual([rsrc_ok], grouputils.get_members(group))
    self.assertEqual(['ID-r1'], grouputils.get_member_refids(group))