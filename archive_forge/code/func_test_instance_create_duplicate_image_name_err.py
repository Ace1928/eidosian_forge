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
def test_instance_create_duplicate_image_name_err(self):
    stack_name = 'test_instance_create_image_name_err_stack'
    tmpl, stack = self._setup_test_stack(stack_name)
    wsp = tmpl.t['Resources']['WebServer']['Properties']
    wsp['ImageId'] = 'CentOS 5.2'
    resource_defns = tmpl.resource_definitions(stack)
    instance = instances.Instance('instance_create_image_err', resource_defns['WebServer'], stack)
    self._mock_get_image_id_fail('CentOS 5.2', glance.client_exception.EntityUniqueMatchNotFound(entity='image', args='CentOS 5.2'))
    self.stub_KeypairConstraint_validate()
    self.stub_SnapshotConstraint_validate()
    self.stub_VolumeConstraint_validate()
    self.stub_FlavorConstraint_validate()
    create = scheduler.TaskRunner(instance.create)
    error = self.assertRaises(exception.ResourceFailure, create)
    self.assertEqual("StackValidationFailed: resources.instance_create_image_err: Property error: WebServer.Properties.ImageId: Error validating value 'CentOS 5.2': No image unique match found for CentOS 5.2.", str(error))