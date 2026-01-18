from pyomo.network import Port, Arc
from pyomo.network.foqus_graph import FOQUSGraph
from pyomo.core import (
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.core.expr import identify_variables
from pyomo.repn import generate_standard_repn
import logging, time
from pyomo.common.dependencies import (
imports_available = networkx_available & numpy_available
def select_tear_mip(self, G, solver, solver_io=None, solver_options={}):
    """
        This finds optimal sets of tear edges based on two criteria.
        The primary objective is to minimize the maximum number of
        times any cycle is broken. The secondary criteria is to
        minimize the number of tears.

        This function creates a MIP problem in Pyomo with a doubly
        weighted objective and solves it with the solver arguments.
        """
    model, bin_list = self.select_tear_mip_model(G)
    from pyomo.environ import SolverFactory
    opt = SolverFactory(solver, solver_io=solver_io)
    if not opt.available(exception_flag=False):
        raise ValueError("Solver '%s' (solver_io=%r) is not available, please pass a different solver" % (solver, solver_io))
    opt.solve(model, **solver_options)
    tset = []
    for i in range(len(bin_list)):
        if bin_list[i].value == 1:
            tset.append(i)
    return tset