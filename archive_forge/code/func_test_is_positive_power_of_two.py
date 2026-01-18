import pickle
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.tests.unit.kernel.test_list_container import (
from pyomo.core.kernel.base import ICategorizedObject, ICategorizedObjectContainer
from pyomo.core.kernel.heterogeneous_container import IHeterogeneousContainer
from pyomo.core.kernel.block import IBlock, block, block_dict, block_list
from pyomo.core.kernel.variable import variable, variable_list
from pyomo.core.kernel.piecewise_library.transforms import (
import pyomo.core.kernel.piecewise_library.transforms as transforms
from pyomo.core.kernel.piecewise_library.transforms_nd import (
import pyomo.core.kernel.piecewise_library.transforms_nd as transforms_nd
import pyomo.core.kernel.piecewise_library.util as util
def test_is_positive_power_of_two(self):
    self.assertEqual(util.is_positive_power_of_two(-8), False)
    self.assertEqual(util.is_positive_power_of_two(-4), False)
    self.assertEqual(util.is_positive_power_of_two(-3), False)
    self.assertEqual(util.is_positive_power_of_two(-2), False)
    self.assertEqual(util.is_positive_power_of_two(-1), False)
    self.assertEqual(util.is_positive_power_of_two(0), False)
    self.assertEqual(util.is_positive_power_of_two(1), True)
    self.assertEqual(util.is_positive_power_of_two(2), True)
    self.assertEqual(util.is_positive_power_of_two(3), False)
    self.assertEqual(util.is_positive_power_of_two(4), True)
    self.assertEqual(util.is_positive_power_of_two(5), False)
    self.assertEqual(util.is_positive_power_of_two(6), False)
    self.assertEqual(util.is_positive_power_of_two(7), False)
    self.assertEqual(util.is_positive_power_of_two(8), True)
    self.assertEqual(util.is_positive_power_of_two(15), False)
    self.assertEqual(util.is_positive_power_of_two(16), True)
    self.assertEqual(util.is_positive_power_of_two(31), False)
    self.assertEqual(util.is_positive_power_of_two(32), True)