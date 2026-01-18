from pyomo.gdp import GDP_Error, Disjunction
from pyomo.gdp.disjunct import _DisjunctData, Disjunct
import pyomo.core.expr as EXPR
from pyomo.core.base.component import _ComponentBase
from pyomo.core import (
from pyomo.core.base.block import _BlockData
from pyomo.common.collections import ComponentMap, ComponentSet, OrderedSet
from pyomo.opt import TerminationCondition, SolverStatus
from weakref import ref as weakref_ref
from collections import defaultdict
import logging
def check_model_algebraic(instance):
    """Checks if there are any active Disjuncts or Disjunctions reachable via
    active Blocks. If there are not, it returns True. If there are, it issues
    a warning detailing where in the model there are remaining non-algebraic
    components, and returns False.

    Parameters
    ----------
    instance: a Model or Block
    """
    disjunction_set = {i for i in instance.component_data_objects(Disjunction, descend_into=(Block, Disjunct), active=None)}
    active_disjunction_set = {i for i in instance.component_data_objects(Disjunction, descend_into=(Block, Disjunct), active=True)}
    disjuncts_in_disjunctions = set()
    for i in disjunction_set:
        disjuncts_in_disjunctions.update(i.disjuncts)
    disjuncts_in_active_disjunctions = set()
    for i in active_disjunction_set:
        disjuncts_in_active_disjunctions.update(i.disjuncts)
    for disjunct in instance.component_data_objects(Disjunct, descend_into=(Block,), descent_order=TraversalStrategy.PostfixDFS):
        if disjunct.transformation_block is not None:
            continue
        elif disjunct.active and _disjunct_not_fixed_true(disjunct) and _disjunct_on_active_block(disjunct):
            if disjunct not in disjuncts_in_disjunctions:
                logger.warning('Disjunct "%s" is currently active, but was not found in any Disjunctions. This is generally an error as the model has not been fully relaxed to a pure algebraic form.' % (disjunct.name,))
                return False
            elif disjunct not in disjuncts_in_active_disjunctions:
                logger.warning('Disjunct "%s" is currently active. While it participates in a Disjunction, that Disjunction is currently deactivated. This is generally an error as the model has not been fully relaxed to a pure algebraic form. Did you deactivate the Disjunction without addressing the individual Disjuncts?' % (disjunct.name,))
                return False
            else:
                logger.warning('Disjunct "%s" is currently active. It must be transformed or deactivated before solving the model.' % (disjunct.name,))
                return False
    for cons in instance.component_data_objects(LogicalConstraint, descend_into=Block, active=True):
        if cons.active:
            logger.warning('LogicalConstraint "%s" is currently active. It must be transformed or deactivated before solving the model.' % cons.name)
            return False
    return True