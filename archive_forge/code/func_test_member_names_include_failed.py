from unittest import mock
from heat.common import grouputils
from heat.common import identifier
from heat.common import template_format
from heat.rpc import client as rpc_client
from heat.tests import common
from heat.tests import utils
def test_member_names_include_failed(self):
    self.list_rsrcs.return_value = self.resources
    self.assertEqual(['B', 'E', 'C', 'A', 'D', 'F'], list(self.insp.member_names(include_failed=True)))
    self.list_rsrcs.assert_called_once_with(self.ctx, dict(self.identity))