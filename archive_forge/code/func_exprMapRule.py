import pyomo.core.expr as EXPR
from pyomo.core import (
from pyomo.core.base.misc import create_name
from pyomo.core.plugins.transform.util import partial
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.core.plugins.transform.util import collectAbstractComponents
import logging
@staticmethod
def exprMapRule(ruleMap, model, ndx=None):
    """Rule intended to return expressions from a lookup table"""
    return ruleMap[ndx]