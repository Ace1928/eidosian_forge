import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.set_utils import (
@m.Block(m.set2, m.time)
def b3(b):
    b.v = Var()
    b.v1 = Var(m.space)

    @b.Block(m.space)
    def b(bb):
        bb.v = Var(m.set)