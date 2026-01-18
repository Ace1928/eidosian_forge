from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
@classmethod
def from_expressions(cls, indices, params, expressions, eps=1e-12, use_cell=False):
    """Converts the expressions into a Jacobian Matrix/const_shift vector and constructs a FixParametricRelations constraint

        The expressions must be a list like object of size 3*N and elements must be ordered as:
        [n_0,i; n_0,j; n_0,k; n_1,i; n_1,j; .... ; n_N-1,i; n_N-1,j; n_N-1,k],
        where i, j, and k are the first, second and third component of the atomic position/lattice
        vector. Currently only linear operations are allowed to be included in the expressions so
        only terms like:
            - const * param_0
            - sqrt[const] * param_1
            - const * param_0 +/- const * param_1 +/- ... +/- const * param_M
        where const is any real number and param_0, param_1, ..., param_M are the parameters passed in
        params, are allowed.

        For example, the fractional atomic position constraints for wurtzite are:
        params = ["z1", "z2"]
        expressions = [
            "1.0/3.0", "2.0/3.0", "z1",
            "2.0/3.0", "1.0/3.0", "0.5 + z1",
            "1.0/3.0", "2.0/3.0", "z2",
            "2.0/3.0", "1.0/3.0", "0.5 + z2",
        ]

        For diamond are:
        params = []
        expressions = [
            "0.0", "0.0", "0.0",
            "0.25", "0.25", "0.25",
        ],

        and for stannite are
        params=["x4", "z4"]
        expressions = [
            "0.0", "0.0", "0.0",
            "0.0", "0.5", "0.5",
            "0.75", "0.25", "0.5",
            "0.25", "0.75", "0.5",
            "x4 + z4", "x4 + z4", "2*x4",
            "x4 - z4", "x4 - z4", "-2*x4",
             "0.0", "-1.0 * (x4 + z4)", "x4 - z4",
             "0.0", "x4 - z4", "-1.0 * (x4 + z4)",
        ]

        Args:
            indices (list of int): indices of the constrained atoms
                (if not None or empty then cell_indices must be None or Empty)
            params (list of str): parameters used in the parametric representation
            expressions (list of str): expressions used to convert from the parametric to the real space
                representation
            eps (float): a small number to compare the similarity of numbers and set the precision used
                to generate the constraint expressions
            use_cell (bool): if True then act on the cell object

        Returns:
            cls(
                indices,
                Jacobian generated from expressions,
                const_shift generated from expressions,
                params,
                eps-12,
                use_cell,
            )
        """
    Jacobian = np.zeros((3 * len(indices), len(params)))
    const_shift = np.zeros(3 * len(indices))
    for expr_ind, expression in enumerate(expressions):
        expression = expression.strip()
        expression = expression.replace('-', '+(-1.0)*')
        if '+' == expression[0]:
            expression = expression[1:]
        elif '(+' == expression[:2]:
            expression = '(' + expression[2:]
        int_fmt_str = '{:0' + str(int(np.ceil(np.log10(len(params) + 1)))) + 'd}'
        param_dct = dict()
        param_map = dict()
        for param_ind, param in enumerate(params):
            param_str = 'param_' + int_fmt_str.format(param_ind)
            param_map[param] = param_str
            param_dct[param_str] = 0.0
        for param in sorted(params, key=lambda s: -1.0 * len(s)):
            expression = expression.replace(param, param_map[param])
        for express_sec in expression.split('+'):
            in_sec = [param in express_sec for param in param_dct]
            n_params_in_sec = len(np.where(np.array(in_sec))[0])
            if n_params_in_sec > 1:
                raise ValueError('The FixParametricRelations expressions must be linear.')
        const_shift[expr_ind] = float(eval_expression(expression, param_dct))
        for param_ind in range(len(params)):
            param_str = 'param_' + int_fmt_str.format(param_ind)
            if param_str not in expression:
                Jacobian[expr_ind, param_ind] = 0.0
                continue
            param_dct[param_str] = 1.0
            test_1 = float(eval_expression(expression, param_dct))
            test_1 -= const_shift[expr_ind]
            Jacobian[expr_ind, param_ind] = test_1
            param_dct[param_str] = 2.0
            test_2 = float(eval_expression(expression, param_dct))
            test_2 -= const_shift[expr_ind]
            if abs(test_2 / test_1 - 2.0) > eps:
                raise ValueError('The FixParametricRelations expressions must be linear.')
            param_dct[param_str] = 0.0
    args = [indices, Jacobian, const_shift, params, eps, use_cell]
    if cls is FixScaledParametricRelations:
        args = args[:-1]
    return cls(*args)