import copy
from unittest import mock
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import vpnservice
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def create_ipsec_site_connection(self):
    self.mockclient.create_ipsec_site_connection.return_value = {'ipsec_site_connection': {'id': 'con123'}}
    snippet = template_format.parse(ipsec_site_connection_template)
    self.stack = utils.parse_stack(snippet)
    resource_defns = self.stack.t.resource_definitions(self.stack)
    return vpnservice.IPsecSiteConnection('ipsec_site_connection', resource_defns['IPsecSiteConnection'], self.stack)