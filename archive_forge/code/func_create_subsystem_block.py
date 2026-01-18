from pyomo.core.base.block import Block
from pyomo.core.base.reference import Reference
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.expression import Expression
from pyomo.core.base.external import ExternalFunction
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.core.expr.numvalue import native_types
def create_subsystem_block(constraints, variables=None, include_fixed=False):
    """This function creates a block to serve as a subsystem with the
    specified variables and constraints. To satisfy certain writers, other
    variables that appear in the constraints must be added to the block as
    well. We call these the "input vars." They may be thought of as
    parameters in the subsystem, but we do not fix them here as it is not
    obvious that this is desired.

    Arguments
    ---------
    constraints: List
        List of Pyomo constraint data objects
    variables: List
        List of Pyomo var data objects
    include_fixed: Bool
        Indicates whether fixed variables should be attached to the block.
        This is useful if they may be unfixed at some point.

    Returns
    -------
    Block containing references to the specified constraints and variables,
    as well as other variables present in the constraints

    """
    if variables is None:
        variables = []
    block = Block(concrete=True)
    block.vars = Reference(variables)
    block.cons = Reference(constraints)
    var_set = ComponentSet(variables)
    input_vars = []
    for con in constraints:
        for var in identify_variables(con.expr, include_fixed=include_fixed):
            if var not in var_set:
                input_vars.append(var)
                var_set.add(var)
    block.input_vars = Reference(input_vars)
    add_local_external_functions(block)
    return block