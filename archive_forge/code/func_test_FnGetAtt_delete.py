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
@mock.patch.object(heat_plugin.HeatClientPlugin, 'get_heat_cfn_url')
def test_FnGetAtt_delete(self, mock_get):
    mock_get.return_value = 'http://server.test:8000/v1'
    stack = self._create_stack(TEMPLATE_CFN_SIGNAL)
    rsrc = stack['signal_handler']
    rsrc.resource_id_set('signal')
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    self.assertIn('http://server.test:8000/v1/signal', rsrc.FnGetAtt('AlarmUrl'))
    scheduler.TaskRunner(rsrc.delete)()
    self.assertIn('http://server.test:8000/v1/signal', rsrc.FnGetAtt('AlarmUrl'))
    self.assertEqual(2, mock_get.call_count)