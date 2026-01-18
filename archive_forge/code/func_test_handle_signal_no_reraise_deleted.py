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
@mock.patch.object(generic_resource.SignalResource, 'handle_signal')
def test_handle_signal_no_reraise_deleted(self, mock_handle):
    test_d = {'Data': 'foo', 'Reason': 'bar', 'Status': 'SUCCESS', 'UniqueId': '123'}
    stack = self._create_stack(TEMPLATE_CFN_SIGNAL)
    mock_handle.side_effect = exception.ResourceNotAvailable(resource_name='test')
    rsrc = stack['signal_handler']
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    with db_api.context_manager.reader.using(stack.context):
        res_obj = stack.context.session.get(models.Resource, rsrc.id)
        res_obj.update({'action': 'DELETE'})
    rsrc._db_res_is_deleted = True
    rsrc._handle_signal(details=test_d)
    mock_handle.assert_called_once_with(test_d)