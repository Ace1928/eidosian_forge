import logging
from pyomo.common.collections import ComponentMap
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.gc_manager import PauseGC
from pyomo.common.modeling import unique_component_name
from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
from pyomo.core import (
from pyomo.core.base import TransformationFactory, Reference
import pyomo.core.expr as EXPR
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.plugins.bigm_mixin import (
from pyomo.gdp.plugins.gdp_to_mip_transformation import GDP_to_MIP_Transformation
from pyomo.gdp.transformed_disjunct import _TransformedDisjunct
from pyomo.gdp.util import is_child_of, _get_constraint_transBlock, _to_dict
from pyomo.core.util import target_list
from pyomo.network import Port
from pyomo.repn import generate_standard_repn
from weakref import ref as weakref_ref, ReferenceType
def get_M_value(self, constraint):
    """Returns the M values used to transform constraint. Return is a tuple:
        (lower_M_value, upper_M_value). Either can be None if constraint does
        not have a lower or upper bound, respectively.

        Parameters
        ----------
        constraint: Constraint, which must be in the subtree of a transformed
                    Disjunct
        """
    transBlock = _get_constraint_transBlock(constraint)
    lower, upper = transBlock.bigm_src[constraint]
    return (lower[0], upper[0])