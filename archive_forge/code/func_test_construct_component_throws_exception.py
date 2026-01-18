from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.environ import (
from pyomo.core.base.set import GlobalSets
def test_construct_component_throws_exception(self):
    with self.assertRaisesRegex(DeveloperError, 'Must specify a component type for class Component'):
        Component()