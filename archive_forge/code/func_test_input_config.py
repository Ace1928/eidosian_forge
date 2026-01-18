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
def test_input_config(self):
    name = 'bar'
    inp = swc_io.InputConfig(name=name, description='test', type='Number', default=0, replace_on_change=True)
    self.assertEqual(0, inp.default())
    self.assertIs(True, inp.replace_on_change())
    self.assertEqual(name, inp.name())
    self.assertEqual({'name': name, 'type': 'Number', 'description': 'test', 'default': 0, 'replace_on_change': True}, inp.as_dict())
    self.assertEqual((name, None), inp.input_data())