from pyomo.core.base.block import _BlockData, declare_custom_block
import pyomo.environ as pyo
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentSet
import logging
def add_subproblem(self, subproblem_fn, subproblem_fn_kwargs, root_eta, subproblem_solver='gurobi_persistent', relax_subproblem_cons=False):
    _rank = np.argmin(self.num_subproblems_by_rank)
    self.num_subproblems_by_rank[_rank] += 1
    self.all_root_etas.append(root_eta)
    if _rank == self.comm.Get_rank():
        self.root_etas.append(root_eta)
        subproblem, complicating_vars_map = subproblem_fn(**subproblem_fn_kwargs)
        self.subproblems.append(subproblem)
        self.complicating_vars_maps.append(complicating_vars_map)
        _setup_subproblem(subproblem, root_vars=[complicating_vars_map[i] for i in self.root_vars if i in complicating_vars_map], relax_subproblem_cons=relax_subproblem_cons)
        self._subproblem_ndx_map[len(self.subproblems) - 1] = self.global_num_subproblems() - 1
        if isinstance(subproblem_solver, str):
            subproblem_solver = pyo.SolverFactory(subproblem_solver)
        self.subproblem_solvers.append(subproblem_solver)
        if isinstance(subproblem_solver, PersistentSolver):
            subproblem_solver.set_instance(subproblem)