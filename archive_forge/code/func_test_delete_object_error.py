import datetime
import json
from unittest import mock
import uuid
from oslo_utils import timeutils
from swiftclient import client as swiftclient_client
from swiftclient import exceptions as swiftclient_exceptions
from testtools import matchers
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import swift
from heat.engine import node_data
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import template as templatem
from heat.tests import common
from heat.tests import utils
@mock.patch.object(swift.SwiftClientPlugin, '_create')
@mock.patch.object(resource.Resource, 'physical_resource_name')
def test_delete_object_error(self, mock_name, mock_swift):
    st = create_stack(swiftsignalhandle_template)
    handle = st['test_wait_condition_handle']
    mock_swift_object = mock.Mock()
    mock_swift.return_value = mock_swift_object
    mock_swift_object.head_account.return_value = {'x-account-meta-temp-url-key': '1234'}
    mock_swift_object.url = 'http://fake-host.com:8080/v1/AUTH_1234'
    obj_name = '%s-%s-abcdefghijkl' % (st.name, handle.name)
    mock_name.return_value = obj_name
    st.create()
    exc = swiftclient_exceptions.ClientException('Overlimit', http_status=413)
    mock_swift_object.delete_object.side_effect = (None, None, None, exc)
    rsrc = st.resources['test_wait_condition_handle']
    exc = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.delete))
    self.assertEqual('ClientException: resources.test_wait_condition_handle: Overlimit: 413', str(exc))