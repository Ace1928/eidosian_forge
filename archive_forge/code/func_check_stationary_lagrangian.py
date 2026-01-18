import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def check_stationary_lagrangian(self, places) -> None:
    L = self.prob.objective.expr
    objective = self.prob.objective
    if objective.NAME == 'minimize':
        L = objective.expr
    else:
        L = -objective.expr
    for con in self.constraints:
        if isinstance(con, (cp.constraints.Inequality, cp.constraints.Equality)):
            dual_var_value = con.dual_value
            prim_var_expr = con.expr
            L = L + cp.scalar_product(dual_var_value, prim_var_expr)
        elif isinstance(con, (cp.constraints.ExpCone, cp.constraints.SOC, cp.constraints.Zero, cp.constraints.NonNeg, cp.constraints.PSD, cp.constraints.PowCone3D, cp.constraints.PowConeND)):
            L = L - cp.scalar_product(con.args, con.dual_value)
        else:
            raise NotImplementedError()
    try:
        g = L.grad
    except TypeError as e:
        assert 'is not subscriptable' in str(e)
        msg = '\n\n            CVXPY problems with `diag` variables are not supported for\n            stationarity checks as of now\n            '
        self.tester.fail(msg)
    bad_norms = []
    'The convention that we follow for construting the Lagrangian is: 1) Move all\n        explicitly passed constraints to the problem (via Problem.constraints) into the\n        Lagrangian --- dLdX == 0 for any such variables 2) Constraints that have\n        implicitly been imposed on variables at the time of declaration via specific\n        flags (e.g.: PSD/symmetric etc.), in such a case we check, `dLdX\\in K^{*}`, where\n        `K` is the convex cone corresponding to the implicit constraint on `X`\n        '
    for opt_var, v in g.items():
        if all((not attr for attr in list(map(lambda x: x[1], opt_var.attributes.items())))):
            "Case when the variable doesn't have any special attributes"
            norm = np.linalg.norm(v.data) / np.sqrt(opt_var.size)
            if norm > 10 ** (-places):
                bad_norms.append((norm, opt_var))
        elif opt_var.is_psd():
            'The PSD cone is self-dual'
            g_bad_mat = cp.Constant(np.reshape(g[opt_var].toarray(), opt_var.shape))
            tmp_con = g_bad_mat >> 0
            dual_cone_violation = tmp_con.residual
            if dual_cone_violation > 10 ** (-places):
                bad_norms.append((dual_cone_violation, opt_var))
        elif opt_var.is_nsd():
            'The NSD cone is also self-dual'
            g_bad_mat = cp.Constant(np.reshape(g[opt_var].toarray(), opt_var.shape))
            tmp_con = g_bad_mat << 0
            dual_cone_violation = tmp_con.residual
            if dual_cone_violation > 10 ** (-places):
                bad_norms.append((dual_cone_violation, opt_var))
        elif opt_var.is_diag():
            "The dual cone to the set of diagonal matrices is the set of\n                        'Hollow' matrices i.e. matrices with diagonal entries zero"
            g_bad_mat = np.reshape(g[opt_var].toarray(), opt_var.shape)
            diag_entries = np.diag(opt_var.value)
            dual_cone_violation = np.linalg.norm(diag_entries) / np.sqrt(opt_var.size)
            if diag_entries > 10 ** (-places):
                bad_norms.append((dual_cone_violation, opt_var))
        elif opt_var.is_symmetric():
            'The dual cone to the set of symmetric matrices is the\n                    set of skew-symmetric matrices, so we check if dLdX \\in\n                    set(skew-symmetric-matrices)\n                    g[opt_var] is the problematic gradient in question'
            g_bad_mat = np.reshape(g[opt_var].toarray(), opt_var.shape)
            mat = g_bad_mat + g_bad_mat.T
            dual_cone_violation = np.linalg.norm(mat) / np.sqrt(opt_var.size)
            if dual_cone_violation > 10 ** (-places):
                bad_norms.append((dual_cone_violation, opt_var))
        elif opt_var.is_nonpos():
            'The cone of matrices with all entries nonpos is self-dual'
            g_bad_mat = cp.Constant(np.reshape(g[opt_var].toarray(), opt_var.shape))
            tmp_con = g_bad_mat <= 0
            dual_cone_violation = np.linalg.norm(tmp_con.residual) / np.sqrt(opt_var.size)
            if dual_cone_violation > 10 ** (-places):
                bad_norms.append((dual_cone_violation, opt_var))
        elif opt_var.is_nonneg():
            'The cone of matrices with all entries nonneg is self-dual'
            g_bad_mat = cp.Constant(np.reshape(g[opt_var].toarray(), opt_var.shape))
            tmp_con = g_bad_mat >= 0
            dual_cone_violation = np.linalg.norm(tmp_con.residual) / np.sqrt(opt_var.size)
            if dual_cone_violation > 10 ** (-places):
                bad_norms.append((dual_cone_violation, opt_var))
    if len(bad_norms):
        msg = f'\n\n        The gradient of Lagrangian with respect to the primal variables\n        is above the threshold of 10^{-places}. The names of the problematic\n        variables and the corresponding gradient norms are as follows:\n            '
        for norm, opt_var in bad_norms:
            msg += f'\n\t\t\t{opt_var.name} : {norm}'
        msg += '\n'
        self.tester.fail(msg)
    pass