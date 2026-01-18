import copy
from unittest import mock
from neutronclient.common import exceptions as q_exceptions
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.aws.ec2 import eip
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def mock_show_floatingip(self):
    self.mock_show_fip.return_value = {'floatingip': {'router_id': None, 'tenant_id': 'e936e6cd3e0b48dcb9ff853a8f253257', 'floating_network_id': 'eeee', 'fixed_ip_address': None, 'floating_ip_address': '11.0.0.1', 'port_id': None, 'id': 'ffff'}}