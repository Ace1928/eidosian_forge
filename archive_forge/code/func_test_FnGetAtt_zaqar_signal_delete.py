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
@mock.patch('zaqarclient.queues.v2.queues.Queue.signed_url')
def test_FnGetAtt_zaqar_signal_delete(self, mock_signed_url):
    stack = self._create_stack(TEMPLATE_ZAQAR_SIGNAL)
    mock_delete = mock.MagicMock()
    rsrc = stack['signal_handler']
    rsrc._delete_zaqar_signal_queue = mock_delete
    stack.create()
    signal = rsrc.FnGetAtt('signal')
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    self.assertEqual('http://localhost:5000/v3', signal['auth_url'])
    self.assertEqual('aprojectid', signal['project_id'])
    self.assertEqual('1234', signal['user_id'])
    self.assertIn('username', signal)
    self.assertIn('password', signal)
    self.assertIn('queue_id', signal)
    scheduler.TaskRunner(rsrc.delete)()
    self.assertEqual('http://localhost:5000/v3', signal['auth_url'])
    self.assertEqual('aprojectid', signal['project_id'])
    self.assertEqual('1234', signal['user_id'])
    self.assertIn('username', signal)
    self.assertIn('password', signal)
    self.assertIn('queue_id', signal)
    mock_delete.assert_called_once_with()