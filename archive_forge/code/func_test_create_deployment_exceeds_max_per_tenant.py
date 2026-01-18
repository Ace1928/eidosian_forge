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
def test_create_deployment_exceeds_max_per_tenant(self):
    cfg.CONF.set_override('max_software_deployments_per_tenant', 0)
    ex = self.assertRaises(dispatcher.ExpectedException, self._create_software_deployment)
    self.assertEqual(exception.RequestLimitExceeded, ex.exc_info[0])
    self.assertIn('You have reached the maximum software deployments per tenant', str(ex.exc_info[1]))