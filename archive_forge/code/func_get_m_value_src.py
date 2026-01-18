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
@deprecated('The get_m_value_src function is deprecated. Use the get_M_value_src function if you need source information or the get_M_value function if you only need values.', version='5.7.1')
def get_m_value_src(self, constraint):
    transBlock = _get_constraint_transBlock(constraint)
    (lower_val, lower_source, lower_key), (upper_val, upper_source, upper_key) = transBlock.bigm_src[constraint]
    if constraint.lower is not None and constraint.upper is not None and (not lower_source is upper_source or not lower_key is upper_key):
        raise GDP_Error('This is why this method is deprecated: The lower and upper M values for constraint %s came from different sources, please use the get_M_value_src method.' % constraint.name)
    if constraint.lower is not None and lower_source is not None:
        return (lower_source, lower_key)
    if constraint.upper is not None and upper_source is not None:
        return (upper_source, upper_key)
    return (lower_val, upper_val)