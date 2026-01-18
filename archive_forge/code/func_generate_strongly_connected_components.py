import logging
from pyomo.core.base.constraint import Constraint
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.subsystems import TemporarySubsystemManager, generate_subsystem_blocks
from pyomo.contrib.incidence_analysis.interface import (
def generate_strongly_connected_components(constraints, variables=None, include_fixed=False):
    """Yield in order ``_BlockData`` that each contain the variables and
    constraints of a single diagonal block in a block lower triangularization
    of the incidence matrix of constraints and variables

    These diagonal blocks correspond to strongly connected components of the
    bipartite incidence graph, projected with respect to a perfect matching
    into a directed graph.

    Parameters
    ----------
    constraints: List of Pyomo constraint data objects
        Constraints used to generate strongly connected components.
    variables: List of Pyomo variable data objects
        Variables that may participate in strongly connected components.
        If not provided, all variables in the constraints will be used.
    include_fixed: Bool
        Indicates whether fixed variables will be included when
        identifying variables in constraints.

    Yields
    ------
    Tuple of ``_BlockData``, list-of-variables
        Blocks containing the variables and constraints of every strongly
        connected component, in a topological order. The variables are the
        "input variables" for that block.

    """
    if variables is None:
        variables = list(_generate_variables_in_constraints(constraints, include_fixed=include_fixed))
    assert len(variables) == len(constraints)
    igraph = IncidenceGraphInterface()
    var_blocks, con_blocks = igraph.block_triangularize(variables=variables, constraints=constraints)
    subsets = [(cblock, vblock) for vblock, cblock in zip(var_blocks, con_blocks)]
    for block, inputs in generate_subsystem_blocks(subsets, include_fixed=include_fixed):
        assert len(block.vars) == len(block.cons)
        yield (block, inputs)