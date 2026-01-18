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
def test_mistral_workflow_refid(self):
    tmpl = template_format.parse(workflow_template)
    stack = utils.parse_stack(tmpl, stack_name='test')
    rsrc = stack['workflow']
    rsrc.uuid = '4c885bde-957e-4758-907b-c188a487e908'
    rsrc.id = 'mockid'
    rsrc.action = 'CREATE'
    self.assertEqual('test-workflow-owevpzgiqw66', rsrc.FnGetRefId())