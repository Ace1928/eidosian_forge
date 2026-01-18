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
def test_create_software_deployment_invalid_stack_user_project_id(self):
    sc_kwargs = {'group': 'Heat::Chef', 'name': 'config_heat', 'config': '...', 'inputs': [{'name': 'mode'}], 'outputs': [{'name': 'endpoint'}], 'options': {}}
    config = self._create_software_config(**sc_kwargs)
    config_id = config['id']
    sd_kwargs = {'config_id': config_id, 'input_values': {'mode': 'standalone'}, 'action': 'INIT', 'status': 'COMPLETE', 'status_reason': '', 'stack_user_project_id': 'a' * 65}
    ex = self.assertRaises(dispatcher.ExpectedException, self._create_software_deployment, **sd_kwargs)
    self.assertEqual(exception.Invalid, ex.exc_info[0])