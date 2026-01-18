from unittest import mock
from neutronclient.common import exceptions
from neutronclient.v2_0 import client as neutronclient
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import firewall
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def create_firewall(self, value_specs=True):
    snippet = template_format.parse(firewall_template)
    self.mockclient.create_firewall.return_value = {'firewall': {'id': '5678'}}
    if not value_specs:
        del snippet['resources']['firewall']['properties']['value_specs']
    self.stack = utils.parse_stack(snippet)
    resource_defns = self.stack.t.resource_definitions(self.stack)
    self.fw_props = snippet['resources']['firewall']['properties']
    return firewall.Firewall('firewall', resource_defns['firewall'], self.stack)