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
def get_disaggregated_var(self, v, disjunct, raise_exception=True):
    """
        Returns the disaggregated variable corresponding to the Var v and the
        Disjunct disjunct.

        If v is a local variable, this method will return v.

        Parameters
        ----------
        v: a Var that appears in a constraint in a transformed Disjunct
        disjunct: a transformed Disjunct in which v appears
        """
    if disjunct._transformation_block is None:
        raise GDP_Error("Disjunct '%s' has not been transformed" % disjunct.name)
    msg = "It does not appear '%s' is a variable that appears in disjunct '%s'" % (v.name, disjunct.name)
    disaggregated_var_map = v.parent_block().private_data().disaggregated_var_map
    if v in disaggregated_var_map[disjunct]:
        return disaggregated_var_map[disjunct][v]
    elif raise_exception:
        raise GDP_Error(msg)