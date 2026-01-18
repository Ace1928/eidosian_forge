from pyomo.common.errors import InfeasibleConstraintException
import pyomo.common.unittest as unittest
from pyomo.environ import Block, ConcreteModel, Constraint, TransformationFactory
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import GDP_Error
def add_three_term_disjunction(self, m, exactly_one=True):
    m.d = Disjunct([1, 2, 3])
    m.disjunction2 = Disjunction(expr=[m.d[1], m.d[2], m.d[3]], xor=exactly_one)