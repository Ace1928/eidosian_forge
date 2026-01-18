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
def test_build_derived_config(self):
    source = {'config': {'foo': {'get_input': 'bar'}}}
    inputs = [swc_io.InputConfig(name='bar', value='baz')]
    result = self.deployment._build_derived_config('CREATE', source, inputs, {})
    self.assertEqual({'foo': 'baz'}, result)