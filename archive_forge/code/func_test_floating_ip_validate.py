import copy
from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from openstack import exceptions
from oslo_utils import excutils
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils
from heat.engine.clients.os import neutron
from heat.engine.hot import functions as hot_funcs
from heat.engine import node_data
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests import utils
def test_floating_ip_validate(self):
    t = template_format.parse(neutron_floating_no_assoc_template)
    stack = utils.parse_stack(t)
    fip = stack['floating_ip']
    self.assertIsNone(fip.validate())
    del t['resources']['floating_ip']['properties']['port_id']
    t['resources']['floating_ip']['properties']['fixed_ip_address'] = '10.0.0.12'
    stack = utils.parse_stack(t)
    fip = stack['floating_ip']
    self.assertRaises(exception.ResourcePropertyDependency, fip.validate)