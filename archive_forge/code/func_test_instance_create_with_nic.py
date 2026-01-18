import copy
import uuid
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine.resources.aws.ec2 import instance as instances
from heat.engine.resources.aws.ec2 import network_interface as net_interfaces
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_instance_create_with_nic(self):
    return_server = self.fc.servers.list()[1]
    instance = self._create_test_instance_with_nic(return_server, 'in_create_wnic')
    self.assertGreater(instance.id, 0)
    expected_ip = return_server.networks['public'][0]
    self.assertEqual(expected_ip, instance.FnGetAtt('PublicIp'))
    self.assertEqual(expected_ip, instance.FnGetAtt('PrivateIp'))
    self.assertEqual(expected_ip, instance.FnGetAtt('PrivateDnsName'))
    self.assertEqual(expected_ip, instance.FnGetAtt('PublicDnsName'))