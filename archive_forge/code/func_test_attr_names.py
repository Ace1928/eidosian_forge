from heat.engine import node_data
from heat.tests import common
def test_attr_names(self):
    nd = make_test_node()
    self.assertEqual({'foo', 'blarg'}, set(nd.attribute_names()))