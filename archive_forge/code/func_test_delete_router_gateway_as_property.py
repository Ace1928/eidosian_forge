import copy
from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import router
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_delete_router_gateway_as_property(self):
    t, stack = self._test_router_with_gateway(for_delete=True)
    rsrc = self.create_router(t, stack, 'router')
    self._assert_mock_call_create_with_router_gw()
    self.assertIsNone(scheduler.TaskRunner(rsrc.delete)())