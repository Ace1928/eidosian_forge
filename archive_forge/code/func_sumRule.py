import pyomo.core.expr as EXPR
from pyomo.core import (
from pyomo.core.base.misc import create_name
from pyomo.core.plugins.transform.util import partial
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.core.plugins.transform.util import collectAbstractComponents
import logging
@staticmethod
def sumRule(attr, vars, model):
    """
        Returns a sum expression.
        """
    return sum((c * model.__getattribute__(attr)[v] for v, c in vars.items()))