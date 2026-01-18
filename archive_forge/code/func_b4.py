import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
@m.Block(m.X, m.Y, m.Z, m.comp)
def b4(b, x, y, z, j):
    b.v = Var()