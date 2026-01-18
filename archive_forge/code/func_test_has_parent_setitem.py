import collections.abc
import pickle
from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.core.kernel.base import ICategorizedObject, ICategorizedObjectContainer
from pyomo.core.kernel.homogeneous_container import IHomogeneousContainer
from pyomo.core.kernel.dict_container import DictContainer
from pyomo.core.kernel.block import block, block_dict
def test_has_parent_setitem(self):
    c = self._container_type()
    c[1] = self._ctype_factory()
    c[1] = c[1]
    with self.assertRaises(ValueError):
        c[2] = c[1]
    d = self._container_type()
    with self.assertRaises(ValueError):
        d[None] = c[1]