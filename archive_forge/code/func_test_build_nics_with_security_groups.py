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
def test_build_nics_with_security_groups(self):
    """Test the security groups can be associated to a new created port.

        Test the security groups defined in heat template can be associated to
        a new created port.
        """
    self.nclient = mock.Mock(spec=neutronclient.Client)
    self.patchobject(neutronclient, 'Client', return_value=self.nclient)
    return_server = self.fc.servers.list()[1]
    instance = self._create_test_instance(return_server, 'build_nics2')
    security_groups = ['security_group_1']
    self._test_security_groups(instance, security_groups)
    security_groups = ['0389f747-7785-4757-b7bb-2ab07e4b09c3']
    self._test_security_groups(instance, security_groups, all_uuids=True)
    security_groups = ['0389f747-7785-4757-b7bb-2ab07e4b09c3', '384ccd91-447c-4d83-832c-06974a7d3d05']
    self._test_security_groups(instance, security_groups, sg='two', all_uuids=True)
    security_groups = ['security_group_1', '384ccd91-447c-4d83-832c-06974a7d3d05']
    self._test_security_groups(instance, security_groups, sg='two')
    security_groups = ['wrong_group_name']
    self._test_security_groups(instance, security_groups, sg='zero', get_secgroup_raises=exception.EntityNotFound)
    security_groups = ['wrong_group_name', '0389f747-7785-4757-b7bb-2ab07e4b09c3']
    self._test_security_groups(instance, security_groups, get_secgroup_raises=exception.EntityNotFound)
    security_groups = ['wrong_group_name', 'security_group_1']
    self._test_security_groups(instance, security_groups, get_secgroup_raises=exception.EntityNotFound)
    security_groups = ['duplicate_group_name', 'security_group_1']
    self._test_security_groups(instance, security_groups, get_secgroup_raises=exception.PhysicalResourceNameAmbiguity)