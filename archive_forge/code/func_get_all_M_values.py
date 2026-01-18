import itertools
import logging
from pyomo.common.collections import ComponentMap
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.gc_manager import PauseGC
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
from pyomo.core.base import Reference, TransformationFactory
import pyomo.core.expr as EXPR
from pyomo.core.util import target_list
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.plugins.bigm_mixin import (
from pyomo.gdp.plugins.gdp_to_mip_transformation import GDP_to_MIP_Transformation
from pyomo.gdp.transformed_disjunct import _TransformedDisjunct
from pyomo.gdp.util import get_gdp_tree, _to_dict
from pyomo.network import Port
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.repn import generate_standard_repn
from weakref import ref as weakref_ref
def get_all_M_values(self, model):
    """Returns a dictionary mapping each constraint, disjunct pair (where
        the constraint is on a disjunct and the disjunct is in the same
        disjunction as that disjunct) to a tuple: (lower_M_value,
        upper_M_value), where either can be None if the constraint does not
        have a lower or upper bound (respectively).

        Parameters
        ----------
        model: A GDP model that has been transformed with multiple-BigM
        """
    all_ms = {}
    for disjunction in model.component_data_objects(Disjunction, active=None, descend_into=(Block, Disjunct), sort=SortComponents.deterministic):
        if disjunction.algebraic_constraint is not None:
            transBlock = disjunction.algebraic_constraint.parent_block()
            if hasattr(transBlock, '_mbm_values'):
                all_ms.update(transBlock._mbm_values)
    return all_ms