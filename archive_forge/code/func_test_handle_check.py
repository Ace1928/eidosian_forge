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
def test_handle_check(self):
    tmpl, stack = self._setup_test_stack('test_instance_check_active')
    res_definitions = tmpl.resource_definitions(stack)
    instance = instances.Instance('instance_create_image', res_definitions['WebServer'], stack)
    instance.client = mock.Mock()
    self.patchobject(nova.NovaClientPlugin, '_check_active', return_value=True)
    self.assertIsNone(instance.handle_check())