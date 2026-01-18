from pyomo.common.deprecation import deprecated
from pyomo.core import (
from pyomo.repn import generate_standard_repn
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.core.plugins.transform.standard_form import StandardForm
from pyomo.core.plugins.transform.util import partial, process_canonical_repn
def objectiveRule(b, model):
    return sum((b[v] * model.vars[v] for v in model.var_set))