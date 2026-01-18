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
def generate_subsystem_blocks(subsystems, include_fixed=False):
    """Generates blocks that contain subsystems of variables and constraints.

    Arguments
    ---------
    subsystems: List of tuples
        Each tuple is a list of constraints then a list of variables
        that will define a subsystem.
    include_fixed: Bool
        Indicates whether to add already fixed variables to the generated
        subsystem blocks.

    Yields
    ------
    "Subsystem blocks" containing the variables and constraints specified
    by each entry in subsystems. Variables in the constraints that are
    not specified are contained in the input_vars component.

    """
    for cons, vars in subsystems:
        block = create_subsystem_block(cons, vars, include_fixed)
        yield (block, list(block.input_vars.values()))