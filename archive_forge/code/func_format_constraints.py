from typing import Tuple
import numpy as np
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone, NonNeg, PowCone3D, Zero
from cvxpy.reductions.cvx_attr2constr import convex_attributes
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.solver import Solver
def format_constraints(self, problem, exp_cone_order):
    """
        Returns a ParamConeProg whose problem data tensors will yield the
        coefficient "A" and offset "b" for the constraint in the following
        formats:
            Linear equations: (A, b) such that A * x + b == 0,
            Linear inequalities: (A, b) such that A * x + b >= 0,
            Second order cone: (A, b) such that A * x + b in SOC,
            Exponential cone: (A, b) such that A * x + b in EXP,
            Semidefinite cone: (A, b) such that A * x + b in PSD,

        The CVXPY standard for the exponential cone is:
            K_e = closure{(x,y,z) |  z >= y * exp(x/y), y>0}.
        Whenever a solver uses this convention, EXP_CONE_ORDER should be
        [0, 1, 2].

        The CVXPY standard for the second order cone is:
            SOC(n) = { x : x[0] >= norm(x[1:n], 2)  }.
        All currently supported solvers use this convention.

        Args:
          problem : ParamConeProg
            The problem that is the provenance of the constraint.
          exp_cone_order: list
            A list indicating how the exponential cone arguments are ordered.

        Returns:
          ParamConeProg with structured A.
        """
    restruct_mat = []
    for constr in problem.constraints:
        total_height = sum([arg.size for arg in constr.args])
        if type(constr) == Zero:
            restruct_mat.append(NegativeIdentityOperator(constr.size))
        elif type(constr) == NonNeg:
            restruct_mat.append(IdentityOperator(constr.size))
        elif type(constr) == SOC:
            assert constr.axis == 0, 'SOC must be lowered to axis == 0'
            t_spacer = ConicSolver.get_spacing_matrix(shape=(total_height, constr.args[0].size), spacing=constr.args[1].shape[0], streak=1, num_blocks=constr.args[0].size, offset=0)
            X_spacer = ConicSolver.get_spacing_matrix(shape=(total_height, constr.args[1].size), spacing=1, streak=constr.args[1].shape[0], num_blocks=constr.args[0].size, offset=1)
            restruct_mat.append(sp.hstack([t_spacer, X_spacer]))
        elif type(constr) == ExpCone:
            arg_mats = []
            for i, arg in enumerate(constr.args):
                space_mat = ConicSolver.get_spacing_matrix(shape=(total_height, arg.size), spacing=len(exp_cone_order) - 1, streak=1, num_blocks=arg.size, offset=exp_cone_order[i])
                arg_mats.append(space_mat)
            restruct_mat.append(sp.hstack(arg_mats))
        elif type(constr) == PowCone3D:
            arg_mats = []
            for i, arg in enumerate(constr.args):
                space_mat = ConicSolver.get_spacing_matrix(shape=(total_height, arg.size), spacing=2, streak=1, num_blocks=arg.size, offset=i)
                arg_mats.append(space_mat)
            restruct_mat.append(sp.hstack(arg_mats))
        elif type(constr) == PSD:
            restruct_mat.append(self.psd_format_mat(constr))
        else:
            raise ValueError('Unsupported constraint type.')
    if restruct_mat:
        restruct_mat = as_block_diag_linear_operator(restruct_mat)
        unspecified, remainder = divmod(problem.A.shape[0] * problem.A.shape[1], restruct_mat.shape[1])
        reshaped_A = problem.A.reshape(restruct_mat.shape[1], unspecified, order='F').tocsr()
        restructured_A = restruct_mat(reshaped_A).tocoo()
        restructured_A.row = restructured_A.row.astype(np.int64)
        restructured_A.col = restructured_A.col.astype(np.int64)
        restructured_A = restructured_A.reshape(np.int64(restruct_mat.shape[0]) * (np.int64(problem.x.size) + 1), problem.A.shape[1], order='F')
    else:
        restructured_A = problem.A
    new_param_cone_prog = ParamConeProg(problem.c, problem.x, restructured_A, problem.variables, problem.var_id_to_col, problem.constraints, problem.parameters, problem.param_id_to_col, P=problem.P, formatted=True)
    return new_param_cone_prog