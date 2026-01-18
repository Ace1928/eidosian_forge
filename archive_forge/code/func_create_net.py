import copy
from neutronclient.common import exceptions as qe
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import net
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def create_net(self, t, stack, resource_name):
    resource_defns = stack.t.resource_definitions(stack)
    rsrc = net.Net('test_net', resource_defns[resource_name], stack)
    scheduler.TaskRunner(rsrc.create)()
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    return rsrc