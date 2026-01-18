import logging
import math
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.collections import ComponentSet
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core import Var, ConstraintList, Expression, Objective
from pyomo.dae import ContinuousSet, DerivativeVar, Integral
from pyomo.dae.misc import generate_finite_elements
from pyomo.dae.misc import generate_colloc_points
from pyomo.dae.misc import expand_components
from pyomo.dae.misc import create_partial_expression
from pyomo.dae.misc import add_discretization_equations
from pyomo.dae.misc import add_continuity_equations
from pyomo.dae.misc import block_fully_discretized
from pyomo.dae.misc import get_index_information
from pyomo.dae.diffvar import DAE_Error
from pyomo.common.config import ConfigBlock, ConfigValue, PositiveInt, In
def calc_adot(cp, order=1):
    a = []
    for i in range(len(cp)):
        ptmp = []
        tmp = 0
        for j in range(len(cp)):
            if j != i:
                row = []
                row.insert(0, 1 / (cp[i] - cp[j]))
                row.insert(1, -cp[j] / (cp[i] - cp[j]))
                ptmp.insert(tmp, row)
                tmp += 1
        p = [1]
        for j in range(len(cp) - 1):
            p = conv(p, ptmp[j])
        pder = numpy.polyder(p, order)
        arow = []
        for j in range(len(cp)):
            arow.append(float(numpy.polyval(pder, cp[j])))
        a.append(arow)
    return a