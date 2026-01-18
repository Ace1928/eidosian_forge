import logging
import textwrap
from math import fabs
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import (
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.preprocessing.util import SuppressConstantObjectiveWarning
from pyomo.core import (
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolverFactory
from pyomo.repn import generate_standard_repn
def detect_effectively_discrete_vars(block, equality_tolerance):
    """Detect effectively discrete variables.

    These continuous variables are the sum of discrete variables.

    """
    effectively_discrete = ComponentMap()
    for constr in block.component_data_objects(Constraint, active=True):
        if constr.lower is None or constr.upper is None:
            continue
        if fabs(value(constr.lower) - value(constr.upper)) > equality_tolerance:
            continue
        if constr.body.polynomial_degree() not in (1, 0):
            continue
        repn = generate_standard_repn(constr.body)
        if len(repn.linear_vars) < 2:
            continue
        non_discrete_vars = list((v for v in repn.linear_vars if v.is_continuous()))
        if len(non_discrete_vars) == 1:
            var = non_discrete_vars[0]
            inducing_constraints = effectively_discrete.get(var, [])
            inducing_constraints.append(constr)
            effectively_discrete[var] = inducing_constraints
    return effectively_discrete