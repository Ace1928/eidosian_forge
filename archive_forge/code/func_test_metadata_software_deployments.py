import datetime
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_messaging.rpc import dispatcher
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from heat.common import crypt
from heat.common import exception
from heat.common import template_format
from heat.db import api as db_api
from heat.engine.clients.os import swift
from heat.engine.clients.os import zaqar
from heat.engine import service
from heat.engine import service_software_config
from heat.engine import software_config_io as swc_io
from heat.objects import resource as resource_objects
from heat.objects import software_config as software_config_object
from heat.objects import software_deployment as software_deployment_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
def test_metadata_software_deployments(self):
    stack_name = 'test_metadata_software_deployments'
    t = template_format.parse(tools.wp_template)
    stack = utils.parse_stack(t, stack_name=stack_name)
    fc = tools.setup_mocks_with_mock(self, stack)
    stack.store()
    stack.create()
    server = stack['WebServer']
    server_id = server.resource_id
    stack_user_project_id = str(uuid.uuid4())
    d1 = self._create_software_deployment(config_group='mygroup', server_id=server_id, config_name='02_second', stack_user_project_id=stack_user_project_id)
    d2 = self._create_software_deployment(config_group='mygroup', server_id=server_id, config_name='01_first', stack_user_project_id=stack_user_project_id)
    d3 = self._create_software_deployment(config_group='myothergroup', server_id=server_id, config_name='03_third', stack_user_project_id=stack_user_project_id)
    metadata = self.engine.metadata_software_deployments(self.ctx, server_id=server_id)
    self.assertEqual(3, len(metadata))
    self.assertEqual('mygroup', metadata[1]['group'])
    self.assertEqual('mygroup', metadata[0]['group'])
    self.assertEqual('myothergroup', metadata[2]['group'])
    self.assertEqual(d1['config_id'], metadata[1]['id'])
    self.assertEqual(d2['config_id'], metadata[0]['id'])
    self.assertEqual(d3['config_id'], metadata[2]['id'])
    self.assertEqual('01_first', metadata[0]['name'])
    self.assertEqual('02_second', metadata[1]['name'])
    self.assertEqual('03_third', metadata[2]['name'])
    rsrcs = resource_objects.Resource.get_all_by_physical_resource_id(self.ctx, server_id)
    self.assertEqual(metadata, rsrcs[0].rsrc_metadata.get('deployments'))
    deployments = self.engine.metadata_software_deployments(self.ctx, server_id=str(uuid.uuid4()))
    self.assertEqual([], deployments)
    ctx = utils.dummy_context(tenant_id=stack_user_project_id)
    metadata = self.engine.metadata_software_deployments(ctx, server_id=server_id)
    self.assertEqual(3, len(metadata))
    ctx = utils.dummy_context(tenant_id=str(uuid.uuid4()))
    metadata = self.engine.metadata_software_deployments(ctx, server_id=server_id)
    self.assertEqual(0, len(metadata))
    obj_conf = self._create_dummy_config_object()
    side_effect = [obj_conf, obj_conf, None]
    self.patchobject(software_config_object.SoftwareConfig, '_from_db_object', side_effect=side_effect)
    metadata = self.engine.metadata_software_deployments(self.ctx, server_id=server_id)
    self.assertEqual(2, len(metadata))
    tools.validate_setup_mocks_with_mock(stack, fc)