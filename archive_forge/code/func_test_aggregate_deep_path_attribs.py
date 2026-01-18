import copy
from unittest import mock
from heat.common import exception
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_chain
from heat.engine import rsrc_defn
from heat.objects import service as service_objects
from heat.tests import common
from heat.tests import utils
def test_aggregate_deep_path_attribs(self):
    """Test attribute aggregation.

        Test attribute aggregation and that we mimic the nested resource's
        attributes.
        """
    chain = self._create_dummy_stack(expect_attrs={'0': 3, '1': 3})
    expected = [3, 3]
    self.assertEqual(expected, chain.FnGetAtt('nested_dict', 'list', 2))