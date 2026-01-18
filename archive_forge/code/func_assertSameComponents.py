from itertools import zip_longest
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
import pyomo.kernel as pmo
from pyomo.util.components import iter_component, rename_components
def assertSameComponents(self, obj, other_obj):
    for i, j in zip_longest(obj, other_obj):
        self.assertEqual(id(i), id(j))