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
def test_characterize_function(self):
    with self.assertRaises(ValueError):
        util.characterize_function([1, 2, -1], [1, 1, 1])
    fc, slopes = util.characterize_function([1, 2, 3], [1, 1, 1])
    self.assertEqual(fc, 1)
    self.assertEqual(slopes, [0, 0])
    fc, slopes = util.characterize_function([1, 2, 3], [1, 0, 1])
    self.assertEqual(fc, 2)
    self.assertEqual(slopes, [-1, 1])
    fc, slopes = util.characterize_function([1, 2, 3], [1, 2, 1])
    self.assertEqual(fc, 3)
    self.assertEqual(slopes, [1, -1])
    fc, slopes = util.characterize_function([1, 1, 2], [1, 2, 1])
    self.assertEqual(fc, 4)
    self.assertEqual(slopes, [None, -1])
    fc, slopes = util.characterize_function([1, 2, 3, 4], [1, 2, 1, 2])
    self.assertEqual(fc, 5)
    self.assertEqual(slopes, [1, -1, 1])