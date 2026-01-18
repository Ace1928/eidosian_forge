from unittest import mock
import yaml
from mistralclient.api import base as mistral_base
from mistralclient.api.v2 import executions
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import mistral as client
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.mistral import workflow
from heat.engine.resources import signal_responder
from heat.engine.resources import stack_user
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_delete_executions_successful(self):
    wf = self._create_resource('workflow')
    self.mistral.executuions.delete.return_value = None
    wf._data = {'executions': '1234,5678'}
    data_delete = self.patchobject(resource.Resource, 'data_delete')
    wf._delete_executions()
    self.assertEqual(2, self.mistral.executions.delete.call_count)
    data_delete.assert_called_once_with('executions')