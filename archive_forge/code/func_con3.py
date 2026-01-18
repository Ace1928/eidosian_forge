import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.initialization import (
@m.fs.Constraint(m.time, m.space)
def con3(fs, t, x):
    if x == m.space.first():
        return Constraint.Skip
    return fs.b2[t, x].v['a'] == 7.0