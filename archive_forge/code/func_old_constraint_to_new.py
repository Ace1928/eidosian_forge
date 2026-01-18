import numpy as np
from ._hessian_update_strategy import BFGS
from ._differentiable_functions import (
from ._optimize import OptimizeWarning
from warnings import warn, catch_warnings, simplefilter, filterwarnings
from scipy.sparse import issparse
def old_constraint_to_new(ic, con):
    """
    Converts old-style constraint dictionaries to new-style constraint objects.
    """
    try:
        ctype = con['type'].lower()
    except KeyError as e:
        raise KeyError('Constraint %d has no type defined.' % ic) from e
    except TypeError as e:
        raise TypeError('Constraints must be a sequence of dictionaries.') from e
    except AttributeError as e:
        raise TypeError("Constraint's type must be a string.") from e
    else:
        if ctype not in ['eq', 'ineq']:
            raise ValueError("Unknown constraint type '%s'." % con['type'])
    if 'fun' not in con:
        raise ValueError('Constraint %d has no function defined.' % ic)
    lb = 0
    if ctype == 'eq':
        ub = 0
    else:
        ub = np.inf
    jac = '2-point'
    if 'args' in con:
        args = con['args']

        def fun(x):
            return con['fun'](x, *args)
        if 'jac' in con:

            def jac(x):
                return con['jac'](x, *args)
    else:
        fun = con['fun']
        if 'jac' in con:
            jac = con['jac']
    return NonlinearConstraint(fun, lb, ub, jac)