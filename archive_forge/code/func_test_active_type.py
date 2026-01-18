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
def test_active_type(self):
    cdict = self._container_type()
    self.assertTrue(isinstance(cdict, ICategorizedObject))
    self.assertTrue(isinstance(cdict, ICategorizedObjectContainer))
    self.assertTrue(isinstance(cdict, IHomogeneousContainer))
    self.assertTrue(isinstance(cdict, DictContainer))
    self.assertTrue(isinstance(cdict, collections.abc.Mapping))
    self.assertTrue(isinstance(cdict, collections.abc.MutableMapping))