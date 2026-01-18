import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
@b1.Block(m.space)
def b_s(b_s):
    b_s.v0 = Var()
    b_s.v1 = Var(m.space)
    b_s.v2 = Var(m.space, m.comp)