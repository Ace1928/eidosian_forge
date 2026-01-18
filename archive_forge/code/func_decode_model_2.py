import logging
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.common.dependencies import networkx_available, matplotlib_available
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import (
from pyomo.contrib.community_detection.detection import (
from pyomo.contrib.community_detection.community_graph import generate_model_graph
from pyomo.solvers.tests.models.LP_unbounded import LP_unbounded
from pyomo.solvers.tests.models.QP_simple import QP_simple
from pyomo.solvers.tests.models.LP_inactive_index import LP_inactive_index
from pyomo.solvers.tests.models.SOS1_simple import SOS1_simple
def decode_model_2():
    model = m = ConcreteModel()
    m.x = Var(RangeSet(1, 7))
    m.c1 = Constraint(expr=m.x[1] + m.x[2] + m.x[3] <= 0)
    m.c2 = Constraint(expr=m.x[1] + 2 * m.x[2] + m.x[3] <= 0)
    m.c3 = Constraint(expr=m.x[3] + m.x[4] + m.x[5] <= 0)
    m.c4 = Constraint(expr=m.x[4] + m.x[5] + m.x[6] + m.x[7] <= 0)
    m.c5 = Constraint(expr=m.x[4] + 2 * m.x[5] + m.x[6] + 0.5 * m.x[7] <= 0)
    m.c6 = Constraint(expr=m.x[4] + m.x[5] + 3 * m.x[6] + m.x[7] <= 0)
    return model