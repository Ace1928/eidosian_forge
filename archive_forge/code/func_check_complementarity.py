import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def check_complementarity(self, places) -> None:
    for con in self.constraints:
        if isinstance(con, (cp.constraints.Inequality, cp.constraints.Equality)):
            comp = cp.scalar_product(con.expr, con.dual_value).value
        elif isinstance(con, (cp.constraints.ExpCone, cp.constraints.SOC, cp.constraints.NonNeg, cp.constraints.Zero, cp.constraints.PSD, cp.constraints.PowCone3D, cp.constraints.PowConeND)):
            comp = cp.scalar_product(con.args, con.dual_value).value
        elif isinstance(con, cp.RelEntrConeQuad) or isinstance(con, cp.OpRelEntrConeQuad):
            msg = '\nDual variables not implemented for quadrature based approximations;' + '\nSkipping complementarity check.'
            warnings.warn(msg)
        else:
            raise ValueError('Unknown constraint type %s.' % type(con))
        self.tester.assertAlmostEqual(comp, 0, places)