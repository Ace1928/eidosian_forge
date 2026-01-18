import collections.abc
import pickle
from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.common.log import LoggingIntercept
from pyomo.core.kernel.base import ICategorizedObject, ICategorizedObjectContainer
from pyomo.core.kernel.homogeneous_container import IHomogeneousContainer
from pyomo.core.kernel.list_container import ListContainer
from pyomo.core.kernel.block import block, block_list
def test_has_parent_insert(self):
    c = self._container_type()
    c.append(self._ctype_factory())
    c.insert(0, self._ctype_factory())
    with self.assertRaises(ValueError):
        c.insert(0, c[0])
    d = []
    d.insert(0, c[0])
    d = self._container_type()
    with self.assertRaises(ValueError):
        d.insert(0, c[0])