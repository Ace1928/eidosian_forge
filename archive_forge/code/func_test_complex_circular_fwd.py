from heat.common import exception
from heat.engine import dependencies
from heat.tests import common
def test_complex_circular_fwd(self):
    d = dependencies.Dependencies([('last', 'e1'), ('last', 'mid1'), ('last', 'mid2'), ('mid1', 'e2'), ('mid1', 'mid3'), ('mid2', 'mid3'), ('mid3', 'e3'), ('e3', 'mid1')])
    self.assertRaises(exception.CircularDependencyException, list, iter(d))