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
def test_mistral_workflow_refid_convergence_cache_data(self):
    tmpl = template_format.parse(workflow_template)
    cache_data = {'workflow': node_data.NodeData.from_dict({'uuid': mock.ANY, 'id': mock.ANY, 'action': 'CREATE', 'status': 'COMPLETE', 'reference_id': 'convg_xyz'})}
    stack = utils.parse_stack(tmpl, stack_name='test', cache_data=cache_data)
    rsrc = stack.defn['workflow']
    self.assertEqual('convg_xyz', rsrc.FnGetRefId())