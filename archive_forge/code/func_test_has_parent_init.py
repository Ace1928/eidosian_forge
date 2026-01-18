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
def test_has_parent_init(self):
    c = self._container_type()
    c[1] = self._ctype_factory()
    with self.assertRaises(ValueError):
        d = self._container_type(c)
    with self.assertRaises(ValueError):
        d = self._container_type(dict(c))