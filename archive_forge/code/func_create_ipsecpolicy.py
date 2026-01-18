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
def create_ipsecpolicy(self):
    self.mockclient.create_ipsecpolicy.return_value = {'ipsecpolicy': {'id': 'ips123'}}
    snippet = template_format.parse(ipsecpolicy_template)
    self.stack = utils.parse_stack(snippet)
    resource_defns = self.stack.t.resource_definitions(self.stack)
    return vpnservice.IPsecPolicy('ipsecpolicy', resource_defns['IPsecPolicy'], self.stack)