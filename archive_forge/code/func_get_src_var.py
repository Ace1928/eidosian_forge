import logging
from collections import defaultdict
from pyomo.common.autoslots import AutoSlots
import pyomo.common.config as cfg
from pyomo.common import deprecated
from pyomo.common.collections import ComponentMap, ComponentSet, DefaultComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.core.expr.numvalue import ZeroConstant
import pyomo.core.expr as EXPR
from pyomo.core.base import TransformationFactory, Reference
from pyomo.core import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.disjunct import _DisjunctData
from pyomo.gdp.plugins.gdp_to_mip_transformation import GDP_to_MIP_Transformation
from pyomo.gdp.transformed_disjunct import _TransformedDisjunct
from pyomo.gdp.util import (
from pyomo.core.util import target_list
from pyomo.util.vars_from_expressions import get_vars_from_components
from weakref import ref as weakref_ref
def get_src_var(self, disaggregated_var):
    """
        Returns the original model variable to which disaggregated_var
        corresponds.

        Parameters
        ----------
        disaggregated_var: a Var that was created by the hull
                           transformation as a disaggregated variable
                           (and so appears on a transformation block
                           of some Disjunct)
        """
    var_map = disaggregated_var.parent_block().private_data()
    if disaggregated_var in var_map.original_var_map:
        return var_map.original_var_map[disaggregated_var]
    raise GDP_Error("'%s' does not appear to be a disaggregated variable" % disaggregated_var.name)