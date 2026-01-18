from pyomo.core.expr import identify_variables, value, differentiate
import logging
import math
def taylor_series_expansion(expr, diff_mode=differentiate.Modes.reverse_numeric, order=1):
    """
    Generate a taylor series approximation for expr.

    Parameters
    ----------
    expr: pyomo.core.expr.numeric_expr.NumericExpression
    diff_mode: pyomo.core.expr.calculus.derivatives.Modes
        The method for differentiation.
    order: The order of the taylor series expansion
        If order is not 1, then symbolic differentiation must
        be used (differentiation.Modes.reverse_sybolic or
        differentiation.Modes.sympy).

    Returns
    -------
    res: pyomo.core.expr.numeric_expr.NumericExpression
    """
    if order < 0:
        raise ValueError('Cannot compute taylor series expansion of order {0}'.format(str(order)))
    if order != 1 and diff_mode is differentiate.Modes.reverse_numeric:
        logger.warning('taylor_series_expansion can only use symbolic differentiation for orders larger than 1')
        diff_mode = differentiate.Modes.reverse_symbolic
    e_vars = list(identify_variables(expr=expr, include_fixed=False))
    res = value(expr)
    if order >= 1:
        derivs = differentiate(expr=expr, wrt_list=e_vars, mode=diff_mode)
        res += sum(((e_vars[i] - e_vars[i].value) * value(derivs[i]) for i in range(len(e_vars))))
    "\n    This last bit of code is just for higher order taylor series expansions.\n    The recursive function _loop modifies derivs in place so that derivs becomes a \n    list of lists of lists... However, _loop is also a generator so that \n    we don't have to loop through it twice. _loop yields two lists. The \n    first is a list of indices corresponding to the first k-1 variables that\n    differentiation is being done with respect to. The second is a list of \n    derivatives. Each entry in this list is the derivative with respect to \n    the first k-1 variables and the kth variable, whose index matches the \n    index in _derivs.\n    "
    if order >= 2:
        for n in range(2, order + 1):
            coef = 1.0 / math.factorial(n)
            for ndx_list, _derivs in _loop(derivs, e_vars, diff_mode, list()):
                tmp = coef
                for ndx in ndx_list:
                    tmp *= e_vars[ndx] - e_vars[ndx].value
                res += tmp * sum(((e_vars[i] - e_vars[i].value) * value(_derivs[i]) for i in range(len(e_vars))))
    return res