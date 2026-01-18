from unittest import mock
from heat.common import exception
from heat.engine.resources.openstack.heat import structured_config as sc
from heat.engine import rsrc_defn
from heat.engine import software_config_io as swc_io
from heat.engine import stack as parser
from heat.engine import template
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import utils
def test_build_derived_config_params_with_empty_config(self):
    source = {}
    source[rpc_api.SOFTWARE_CONFIG_INPUTS] = []
    source[rpc_api.SOFTWARE_CONFIG_OUTPUTS] = []
    result = self.deployment._build_derived_config_params('CREATE', source)
    self.assertEqual('Heat::Ungrouped', result['group'])
    self.assertEqual({}, result['config'])
    self.assertEqual(self.deployment.physical_resource_name(), result['name'])
    self.assertIn({'name': 'bar', 'type': 'String', 'value': 'baz'}, result['inputs'])
    self.assertIsNone(result['options'])
    self.assertEqual([], result['outputs'])