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
def test__raise_if_duplicate_positive(self):
    self.assertRaises(exception.PhysicalResourceExists, extrarouteset._raise_if_duplicate, {'router': {'routes': [{'destination': 'dst1', 'nexthop': 'hop1'}]}}, [{'destination': 'dst1', 'nexthop': 'hop1'}])