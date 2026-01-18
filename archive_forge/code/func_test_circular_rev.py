from heat.common import exception
from heat.engine import dependencies
from heat.tests import common
def test_circular_rev(self):
    d = dependencies.Dependencies([('first', 'second'), ('second', 'third'), ('third', 'first')])
    self.assertRaises(exception.CircularDependencyException, list, reversed(d))