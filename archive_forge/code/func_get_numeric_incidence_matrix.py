import enum
import textwrap
from pyomo.core.base.block import _BlockData
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.objective import Objective
from pyomo.core.expr import EqualityExpression
from pyomo.util.subsystems import create_subsystem_block
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.dependencies import (
from pyomo.common.deprecation import deprecated
from pyomo.contrib.incidence_analysis.config import get_config_from_kwds
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.connected import get_independent_submatrices
from pyomo.contrib.incidence_analysis.triangularize import (
from pyomo.contrib.incidence_analysis.dulmage_mendelsohn import (
from pyomo.contrib.incidence_analysis.incidence import get_incident_variables
from pyomo.contrib.pynumero.asl import AmplInterface
def get_numeric_incidence_matrix(variables, constraints):
    """Return the "numeric incidence matrix" (Jacobian) of Pyomo variables
    and constraints.

    Each matrix value is the derivative of a constraint body with respect
    to a variable. Rows correspond to constraints and columns correspond to
    variables. Entries are included even if the value of the derivative is
    zero.
    Only active constraints and unfixed variables that participate in these
    constraints are included.

    Parameters
    ---------
    variables: List of Pyomo VarData objects
    constraints: List of Pyomo ConstraintData objects

    Returns
    -------
    ``scipy.sparse.coo_matrix``
        COO matrix. Rows are indices into the user-provided list of constraints,
        columns are indices into the user-provided list of variables.

    """
    comps = list(variables) + list(constraints)
    _check_unindexed(comps)
    block = create_subsystem_block(constraints, variables)
    block._obj = Objective(expr=0)
    nlp = pyomo_nlp.PyomoNLP(block)
    return nlp.extract_submatrix_jacobian(variables, constraints)