from unittest import mock
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine import scheduler
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
def test_launch_config_refid_convergence_cache_data(self):
    t = template_format.parse(inline_templates.as_template)
    cache_data = {'LaunchConfig': node_data.NodeData.from_dict({'uuid': mock.ANY, 'id': mock.ANY, 'action': 'CREATE', 'status': 'COMPLETE', 'reference_id': 'convg_xyz'})}
    stack = utils.parse_stack(t, params=inline_templates.as_params, cache_data=cache_data)
    rsrc = stack.defn['LaunchConfig']
    self.assertEqual('convg_xyz', rsrc.FnGetRefId())