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
@mock.patch('swiftclient.client.Connection.put_container')
@mock.patch('swiftclient.client.Connection.put_object')
@mock.patch.object(swift.SwiftClientPlugin, 'get_temp_url')
def test_FnGetAtt_swift_signal_is_cached(self, mock_get_url, mock_put_object, mock_put_container):
    mock_get_url.return_value = 'http://192.0.2.1/v1/AUTH_aprojectid/foo/bar'
    stack = self._create_stack(TEMPLATE_SWIFT_SIGNAL)
    rsrc = stack['signal_handler']
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    first_url = rsrc.FnGetAtt('AlarmUrl')
    second_url = rsrc.FnGetAtt('AlarmUrl')
    self.assertEqual(first_url, second_url)
    self.assertEqual(1, mock_put_container.call_count)
    self.assertEqual(1, mock_put_object.call_count)
    self.assertEqual(1, mock_get_url.call_count)