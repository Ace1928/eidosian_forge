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
def decode_model_1():
    model = m = ConcreteModel()
    m.x1 = Var(initialize=-3)
    m.x2 = Var(initialize=-1)
    m.x3 = Var(initialize=-3)
    m.x4 = Var(initialize=-1)
    m.c1 = Constraint(expr=m.x1 + m.x2 <= 0)
    m.c2 = Constraint(expr=m.x1 - 3 * m.x2 <= 0)
    m.c3 = Constraint(expr=m.x2 + m.x3 + 4 * m.x4 ** 2 == 0)
    m.c4 = Constraint(expr=m.x3 + m.x4 <= 0)
    m.c5 = Constraint(expr=m.x3 ** 2 + m.x4 ** 2 - 10 == 0)
    return model