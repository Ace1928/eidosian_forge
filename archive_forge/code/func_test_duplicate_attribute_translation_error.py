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
def test_duplicate_attribute_translation_error(self):
    tmpl = template_format.parse(workflow_template_duplicate_polices)
    stack = utils.parse_stack(tmpl)
    rsrc_defns = stack.t.resource_definitions(stack)['workflow']
    workflow_rsrc = workflow.Workflow('workflow', rsrc_defns, stack)
    ex = self.assertRaises(exception.StackValidationFailed, workflow_rsrc.validate)
    error_msg = 'Cannot define the following properties at the same time: tasks.retry, tasks.policies.retry'
    self.assertIn(error_msg, str(ex))