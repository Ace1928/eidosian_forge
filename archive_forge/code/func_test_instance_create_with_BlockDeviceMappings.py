import copy
from unittest import mock
import uuid
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import cinder
from heat.engine.clients.os import glance
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine.clients import progress
from heat.engine import environment
from heat.engine import resource
from heat.engine.resources.aws.ec2 import instance as instances
from heat.engine.resources import scheduler_hints as sh
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_instance_create_with_BlockDeviceMappings(self):
    return_server = self.fc.servers.list()[4]
    instance = self._create_test_instance(return_server, 'create_with_bdm')
    self.assertGreater(instance.id, 0)
    expected_ip = return_server.networks['public'][0]
    expected_az = getattr(return_server, 'OS-EXT-AZ:availability_zone')
    self.assertEqual(expected_ip, instance.FnGetAtt('PublicIp'))
    self.assertEqual(expected_ip, instance.FnGetAtt('PrivateIp'))
    self.assertEqual(expected_ip, instance.FnGetAtt('PublicDnsName'))
    self.assertEqual(expected_ip, instance.FnGetAtt('PrivateDnsName'))
    self.assertEqual(expected_az, instance.FnGetAtt('AvailabilityZone'))