from unittest import mock
from neutronclient.v2_0 import client as neutronclient
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import extraroute
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def create_extraroute(self, t, stack, resource_name, properties=None):
    properties = properties or {}
    t['Resources'][resource_name]['Properties'] = properties
    rsrc = extraroute.ExtraRoute(resource_name, stack.t.resource_definitions(stack)[resource_name], stack)
    scheduler.TaskRunner(rsrc.create)()
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    return rsrc