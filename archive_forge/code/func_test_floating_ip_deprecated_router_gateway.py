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
def test_floating_ip_deprecated_router_gateway(self):
    t = template_format.parse(neutron_floating_template_deprecated)
    del t['resources']['router_interface']
    self._test_floating_ip(t, r_iface=False)