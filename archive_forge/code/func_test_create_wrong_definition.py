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
def test_create_wrong_definition(self):
    tmpl = template_format.parse(workflow_template)
    stack = utils.parse_stack(tmpl)
    rsrc_defns = stack.t.resource_definitions(stack)['workflow']
    wf = workflow.Workflow('workflow', rsrc_defns, stack)
    self.mistral.workflows.create.side_effect = Exception('boom!')
    exc = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(wf.create))
    expected_state = (wf.CREATE, wf.FAILED)
    self.assertEqual(expected_state, wf.state)
    self.assertIn('Exception: resources.workflow: boom!', str(exc))