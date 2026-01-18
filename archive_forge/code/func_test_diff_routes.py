import copy
from oslo_log import log as logging
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import extrarouteset
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
from neutronclient.common import exceptions as ncex
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
def test_diff_routes(self):
    old = [{'destination': '10.0.1.0/24', 'nexthop': '10.0.0.11'}, {'destination': '10.0.2.0/24', 'nexthop': '10.0.0.12'}]
    new = [{'destination': '10.0.1.0/24', 'nexthop': '10.0.0.11'}, {'destination': '10.0.3.0/24', 'nexthop': '10.0.0.13'}]
    add = extrarouteset._set_to_routes(extrarouteset._routes_to_set(new) - extrarouteset._routes_to_set(old))
    remove = extrarouteset._set_to_routes(extrarouteset._routes_to_set(old) - extrarouteset._routes_to_set(new))
    self.assertEqual([{'destination': '10.0.3.0/24', 'nexthop': '10.0.0.13'}], add)
    self.assertEqual([{'destination': '10.0.2.0/24', 'nexthop': '10.0.0.12'}], remove)