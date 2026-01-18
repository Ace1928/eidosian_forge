from heat.common import exception
from heat.engine import dependencies
from heat.tests import common
def test_disjoint(self):
    d = dependencies.Dependencies([('1', None), ('2', None)])
    li = list(iter(d))
    self.assertEqual(2, len(li))
    self.assertIn('1', li)
    self.assertIn('2', li)