import copy
from unittest import mock
from heat.common import exception
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_chain
from heat.engine import rsrc_defn
from heat.objects import service as service_objects
from heat.tests import common
from heat.tests import utils
def test_aggregate_outputs_no_path(self):
    """Test outputs aggregation with missing path."""
    chain = self._create_dummy_stack()
    self.assertRaises(exception.InvalidTemplateAttribute, chain.FnGetAtt, 'attributes')