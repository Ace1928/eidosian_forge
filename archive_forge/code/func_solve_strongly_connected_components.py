import logging
from pyomo.core.base.constraint import Constraint
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.subsystems import TemporarySubsystemManager, generate_subsystem_blocks
from pyomo.contrib.incidence_analysis.interface import (
def solve_strongly_connected_components(block, solver=None, solve_kwds=None, calc_var_kwds=None):
    """Solve a square system of variables and equality constraints by
    solving strongly connected components individually.

    Strongly connected components (of the directed graph of constraints
    obtained from a perfect matching of variables and constraints) are
    the diagonal blocks in a block triangularization of the incidence
    matrix, so solving the strongly connected components in topological
    order is sufficient to solve the entire block.

    One-by-one blocks are solved using Pyomo's
    calculate_variable_from_constraint function, while higher-dimension
    blocks are solved using the user-provided solver object.

    Parameters
    ----------
    block: Pyomo Block
        The Pyomo block whose variables and constraints will be solved
    solver: Pyomo solver object
        The solver object that will be used to solve strongly connected
        components of size greater than one constraint. Must implement
        a solve method.
    solve_kwds: Dictionary
        Keyword arguments for the solver's solve method
    calc_var_kwds: Dictionary
        Keyword arguments for calculate_variable_from_constraint

    Returns
    -------
    List of results objects returned by each call to solve

    """
    if solve_kwds is None:
        solve_kwds = {}
    if calc_var_kwds is None:
        calc_var_kwds = {}
    igraph = IncidenceGraphInterface(block, active=True, include_fixed=False, include_inequality=False)
    constraints = igraph.constraints
    variables = igraph.variables
    res_list = []
    log_blocks = _log.isEnabledFor(logging.DEBUG)
    for scc, inputs in generate_strongly_connected_components(constraints, variables):
        with TemporarySubsystemManager(to_fix=inputs):
            N = len(scc.vars)
            if N == 1:
                if log_blocks:
                    _log.debug(f'Solving 1x1 block: {scc.cons[0].name}.')
                results = calculate_variable_from_constraint(scc.vars[0], scc.cons[0], **calc_var_kwds)
                res_list.append(results)
            else:
                if solver is None:
                    var_names = [var.name for var in scc.vars.values()][:10]
                    con_names = [con.name for con in scc.cons.values()][:10]
                    raise RuntimeError('An external solver is required if block has strongly\nconnected components of size greater than one (is not a DAG).\nGot an SCC of size %sx%s including components:\n%s\n%s' % (N, N, var_names, con_names))
                if log_blocks:
                    _log.debug(f'Solving {N}x{N} block.')
                results = solver.solve(scc, **solve_kwds)
                res_list.append(results)
    return res_list