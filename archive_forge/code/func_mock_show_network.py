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
def mock_show_network(self):
    vpc_name = utils.PhysName('test_stack', 'the_vpc')
    self.mock_show_net.return_value = {'network': {'status': 'BUILD', 'subnets': [], 'name': vpc_name, 'admin_state_up': False, 'shared': False, 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'id': '22c26451-cf27-4d48-9031-51f5e397b84e'}}