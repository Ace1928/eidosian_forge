import datetime
from unittest import mock
from urllib import parse as urlparse
from keystoneauth1 import exceptions as kc_exceptions
from oslo_utils import timeutils
from heat.common import exception
from heat.common import template_format
from heat.db import api as db_api
from heat.db import models
from heat.engine.clients.os import heat_plugin
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine.clients.os import swift
from heat.engine import scheduler
from heat.engine import stack as stk
from heat.engine import template
from heat.objects import resource_data as resource_data_object
from heat.tests import common
from heat.tests import generic_resource
from heat.tests import utils
@mock.patch.object(heat_plugin.HeatClientPlugin, 'get_heat_url')
def test_FnGetAtt_heat_signal_delete(self, mock_get):
    mock_get.return_value = 'http://server.test:8004/v1'
    stack = self._create_stack(TEMPLATE_HEAT_TEMPLATE_SIGNAL)
    rsrc = stack['signal_handler']
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)

    def validate_signal():
        signal = rsrc.FnGetAtt('signal')
        self.assertEqual('http://localhost:5000/v3', signal['auth_url'])
        self.assertEqual('aprojectid', signal['project_id'])
        self.assertEqual('1234', signal['user_id'])
        self.assertIn('username', signal)
        self.assertIn('password', signal)
    validate_signal()
    scheduler.TaskRunner(rsrc.delete)()
    validate_signal()
    self.assertEqual(2, mock_get.call_count)