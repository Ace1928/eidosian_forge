import copy
from unittest import mock
from heat.common import exception
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_chain
from heat.engine import rsrc_defn
from heat.objects import service as service_objects
from heat.tests import common
from heat.tests import utils
def test_child_template_default_concurrent(self):
    tmpl_def = copy.deepcopy(TEMPLATE)
    tmpl_def['resources']['test-chain']['properties'].pop('concurrent')
    chain = self._create_chain(tmpl_def)
    child_template = chain.child_template()
    tmpl = child_template.t
    resource = tmpl['resources']['0']
    self.assertNotIn('depends_on', resource)
    resource = tmpl['resources']['1']
    self.assertEqual(['0'], resource['depends_on'])