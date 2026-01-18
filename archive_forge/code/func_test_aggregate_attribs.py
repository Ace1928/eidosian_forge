import copy
from unittest import mock
from heat.common import exception
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_chain
from heat.engine import rsrc_defn
from heat.objects import service as service_objects
from heat.tests import common
from heat.tests import utils
def test_aggregate_attribs(self):
    """Test attribute aggregation.

        Test attribute aggregation and that we mimic the nested resource's
        attributes.
        """
    chain = self._create_dummy_stack()
    expected = ['0', '1']
    self.assertEqual(expected, chain.FnGetAtt('foo'))
    self.assertEqual(expected, chain.FnGetAtt('Foo'))