from sympy.core.backend import diff, zeros, Matrix, eye, sympify
from sympy.core.sorting import default_sort_key
from sympy.physics.vector import dynamicsymbols, ReferenceFrame
from sympy.physics.mechanics.method import _Methods
from sympy.physics.mechanics.functions import (
from sympy.physics.mechanics.linearize import Linearizer
from sympy.utilities.iterables import iterable
def solve_multipliers(self, op_point=None, sol_type='dict'):
    """Solves for the values of the lagrange multipliers symbolically at
        the specified operating point.

        Parameters
        ==========

        op_point : dict or iterable of dicts, optional
            Point at which to solve at. The operating point is specified as
            a dictionary or iterable of dictionaries of {symbol: value}. The
            value may be numeric or symbolic itself.

        sol_type : str, optional
            Solution return type. Valid options are:
            - 'dict': A dict of {symbol : value} (default)
            - 'Matrix': An ordered column matrix of the solution
        """
    k = len(self.lam_vec)
    if k == 0:
        raise ValueError('System has no lagrange multipliers to solve for.')
    if isinstance(op_point, dict):
        op_point_dict = op_point
    elif iterable(op_point):
        op_point_dict = {}
        for op in op_point:
            op_point_dict.update(op)
    elif op_point is None:
        op_point_dict = {}
    else:
        raise TypeError('op_point must be either a dictionary or an iterable of dictionaries.')
    mass_matrix = self.mass_matrix.col_join(-self.lam_coeffs.row_join(zeros(k, k)))
    force_matrix = self.forcing.col_join(self._f_cd)
    mass_matrix = msubs(mass_matrix, op_point_dict)
    force_matrix = msubs(force_matrix, op_point_dict)
    sol_list = mass_matrix.LUsolve(-force_matrix)[-k:]
    if sol_type == 'dict':
        return dict(zip(self.lam_vec, sol_list))
    elif sol_type == 'Matrix':
        return Matrix(sol_list)
    else:
        raise ValueError('Unknown sol_type {:}.'.format(sol_type))