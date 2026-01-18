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
def mock_list_ports(self):
    self.mock_list_ports.return_value = {'ports': [{'status': 'DOWN', 'binding:host_id': 'null', 'name': 'wp-NIC-yu7fc7l4g5p6', 'admin_state_up': True, 'network_id': '22c26451-cf27-4d48-9031-51f5e397b84e', 'tenant_id': 'ecf538ec1729478fa1f97f1bf4fdcf7b', 'binding:vif_type': 'ovs', 'device_owner': '', 'binding:capabilities': {'port_filter': True}, 'mac_address': 'fa:16:3e:62:2d:4f', 'fixed_ips': [{'subnet_id': 'mysubnetid-70ec', 'ip_address': '192.168.9.2'}], 'id': 'a000228d-b40b-4124-8394-a4082ae1b76b', 'security_groups': ['5c6f529d-3186-4c36-84c0-af28b8daac7b'], 'device_id': ''}]}