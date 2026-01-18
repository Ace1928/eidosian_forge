from unittest import mock
from oslo_config import cfg
from oslo_context import context as oslo_context
from testtools import matchers
from neutron_lib import context
from neutron_lib.tests import _base
def test_neutron_context_create_with_request_id(self):
    ctx = context.Context('user_id', 'tenant_id', request_id='req_id_xxx')
    self.assertEqual('req_id_xxx', ctx.request_id)