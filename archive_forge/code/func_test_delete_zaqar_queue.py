import contextlib
import copy
import re
from unittest import mock
import uuid
from oslo_serialization import jsonutils
from heat.common import exception as exc
from heat.common.i18n import _
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine.clients.os import swift
from heat.engine.clients.os import zaqar
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.heat import software_deployment as sd
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
@mock.patch.object(zaqar.ZaqarClientPlugin, 'create_for_tenant')
def test_delete_zaqar_queue(self, zcc):
    queue_id = str(uuid.uuid4())
    dep_data = {'password': 'password', 'zaqar_signal_queue_id': queue_id}
    self._create_stack(self.template_zaqar_signal)
    self.deployment.data_delete = mock.MagicMock()
    self.deployment.data = mock.Mock(return_value=dep_data)
    zc = mock.MagicMock()
    zcc.return_value = zc
    self.deployment.id = 23
    self.deployment.uuid = str(uuid.uuid4())
    self.deployment._delete_zaqar_signal_queue()
    zc.queue.assert_called_once_with(queue_id)
    self.assertTrue(zc.queue(self.deployment.uuid).delete.called)
    self.assertEqual([mock.call('zaqar_signal_queue_id')], self.deployment.data_delete.mock_calls)
    zaqar_exc = zaqar.ZaqarClientPlugin.exceptions_module
    zc.queue.delete.side_effect = zaqar_exc.ResourceNotFound()
    self.deployment._delete_zaqar_signal_queue()
    self.assertEqual([mock.call('zaqar_signal_queue_id'), mock.call('zaqar_signal_queue_id')], self.deployment.data_delete.mock_calls)
    dep_data.pop('zaqar_signal_queue_id')
    self.deployment.physical_resource_name = mock.Mock()
    self.deployment._delete_zaqar_signal_queue()
    self.assertEqual(2, len(self.deployment.data_delete.mock_calls))