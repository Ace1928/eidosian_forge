from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import zip
from builtins import map
from builtins import range
import copy
import weakref
import math
from math import isnan, isinf
import random
import sys
import uncertainties.core as uncert_core
from uncertainties.core import ufloat, AffineScalarFunc, ufloat_fromstr
from uncertainties import umath
def compare_derivatives(func, numerical_derivatives, num_args_list=None):
    """
    Checks the derivatives of a function 'func' (as returned by the
    wrap() wrapper), by comparing them to the
    'numerical_derivatives' functions.

    Raises a DerivativesDiffer exception in case of problem.

    These functions all take the number of arguments listed in
    num_args_list.  If num_args is None, it is automatically obtained.

    Tests are done on random arguments.
    """
    try:
        funcname = func.name
    except AttributeError:
        funcname = func.__name__
    if not num_args_list:
        num_args_table = {'atanh': [1], 'log': [1, 2]}
        if funcname in num_args_table:
            num_args_list = num_args_table[funcname]
        else:
            num_args_list = []
            for num_args in range(10):
                try:
                    func(*(1,) * num_args)
                except TypeError:
                    pass
                else:
                    num_args_list.append(num_args)
            if not num_args_list:
                raise Exception("Can't find a reasonable number of arguments for function '%s'." % funcname)
    for num_args in num_args_list:
        integer_arg_nums = set()
        if funcname == 'ldexp':
            integer_arg_nums.add(1)
        while True:
            try:
                args = []
                for arg_num in range(num_args):
                    if arg_num in integer_arg_nums:
                        args.append(random.choice(range(-10, 10)))
                    else:
                        args.append(uncert_core.Variable(random.random() * 4 - 2, 0))
                args_scalar = [uncert_core.nominal_value(v) for v in args]
                func_approx = func(*args)
                if isinstance(func_approx, AffineScalarFunc):
                    for arg_num, (arg, numerical_deriv) in enumerate(zip(args, numerical_derivatives)):
                        if isinstance(arg, int):
                            continue
                        fixed_deriv_value = func_approx.derivatives[arg]
                        num_deriv_value = numerical_deriv(*args_scalar)
                        print('Testing derivative #%d of %s at %s' % (arg_num, funcname, args_scalar))
                        if not numbers_close(fixed_deriv_value, num_deriv_value, 0.0001):
                            if not isnan(func_approx):
                                raise DerivativesDiffer("Derivative #%d of function '%s' may be wrong: at args = %s, value obtained = %.16f, while numerical approximation = %.16f." % (arg_num, funcname, args, fixed_deriv_value, num_deriv_value))
            except ValueError as err:
                if str(err).startswith('factorial'):
                    integer_arg_nums = set([0])
                continue
            except TypeError as err:
                if len(integer_arg_nums) == num_args:
                    raise Exception('Incorrect testing procedure: unable to find correct argument values for %s: %s' % (funcname, err))
                integer_arg_nums.add(random.choice(range(num_args)))
            else:
                break