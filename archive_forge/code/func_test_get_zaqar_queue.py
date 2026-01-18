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
def test_get_zaqar_queue(self):
    dep_data = {}
    zc = mock.MagicMock()
    zcc = self.patch('heat.engine.clients.os.zaqar.ZaqarClientPlugin.create_for_tenant')
    zcc.return_value = zc
    mock_queue = mock.MagicMock()
    zc.queue.return_value = mock_queue
    signed_data = {'signature': 'hi', 'expires': 'later'}
    mock_queue.signed_url.return_value = signed_data
    self._create_stack(self.template_zaqar_signal)

    def data_set(key, value, redact=False):
        dep_data[key] = value
    self.deployment.data_set = data_set
    self.deployment.data = mock.Mock(return_value=dep_data)
    self.deployment.id = 23
    self.deployment.uuid = str(uuid.uuid4())
    self.deployment.action = self.deployment.CREATE
    queue_id = self.deployment._get_zaqar_signal_queue_id()
    self.assertEqual(queue_id, dep_data['zaqar_signal_queue_id'])
    self.assertEqual(jsonutils.dumps(signed_data), dep_data['zaqar_queue_signed_url_data'])
    self.assertEqual(queue_id, self.deployment._get_zaqar_signal_queue_id())