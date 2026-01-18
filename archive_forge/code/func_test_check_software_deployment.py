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
def test_check_software_deployment(self):
    deployment_id = str(uuid.uuid4())
    ex = self.assertRaises(dispatcher.ExpectedException, self.engine.check_software_deployment, self.ctx, deployment_id, 10)
    self.assertEqual(exception.NotFound, ex.exc_info[0])
    deployment = self._create_software_deployment()
    self.assertIsNotNone(deployment)
    deployment_id = deployment['id']
    self.assertEqual(deployment, self.engine.check_software_deployment(self.ctx, deployment_id, 10))