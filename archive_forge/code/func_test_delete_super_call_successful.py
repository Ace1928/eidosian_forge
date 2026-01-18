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
def test_delete_super_call_successful(self):
    wf = self._create_resource('workflow')
    scheduler.TaskRunner(wf.delete)()
    self.assertEqual((wf.DELETE, wf.COMPLETE), wf.state)
    self.assertEqual(1, self.mistral.workflows.delete.call_count)