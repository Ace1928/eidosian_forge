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
def test_instance_create_unexpected_status(self):
    return_server = self.fc.servers.list()[1]
    instance = self._create_test_instance(return_server, 'test_instance_create')
    creator = progress.ServerCreateProgress(instance.resource_id)
    self.fc.servers.get = mock.Mock(return_value=return_server)
    return_server.status = 'BOGUS'
    e = self.assertRaises(exception.ResourceUnknownStatus, instance.check_create_complete, (creator, None))
    self.assertEqual('Instance is not active - Unknown status BOGUS due to "Unknown"', str(e))
    self.fc.servers.get.assert_called_once_with(instance.resource_id)