import logging
from pyomo.common.collections import ComponentSet, Bunch
from pyomo.core import Block, Constraint, Var
import pyomo.core.expr as EXPR
from pyomo.gdp import Disjunct, Disjunction
def log_model_size_report(model, logger=default_logger):
    """Generate a report logging the model size."""
    logger.info(build_model_size_report(model))