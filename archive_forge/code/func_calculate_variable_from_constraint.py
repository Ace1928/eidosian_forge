from pyomo.common.errors import IterationLimitError
from pyomo.common.numeric_types import native_numeric_types, native_complex_types, value
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.base.constraint import Constraint, _ConstraintData
import logging
def calculate_variable_from_constraint(variable, constraint, eps=1e-08, iterlim=1000, linesearch=True, alpha_min=1e-08, diff_mode=None):
    """Calculate the variable value given a specified equality constraint

    This function calculates the value of the specified variable
    necessary to make the provided equality constraint feasible
    (assuming any other variables values are fixed).  The method first
    attempts to solve for the variable value assuming it appears
    linearly in the constraint.  If that doesn't converge the constraint
    residual, it falls back on Newton's method using exact (symbolic)
    derivatives.

    Notes
    -----
    This is an unconstrained solver and is NOT guaranteed to respect the
    variable bounds or domain.  The solver may leave the variable value
    in an infeasible state (outside the declared bounds or domain bounds).

    Parameters:
    -----------
    variable: :py:class:`_VarData`
        The variable to solve for
    constraint: :py:class:`_ConstraintData` or relational expression or `tuple`
        The equality constraint to use to solve for the variable value.
        May be a `ConstraintData` object or any valid argument for
        ``Constraint(expr=<>)`` (i.e., a relational expression or 2- or
        3-tuple)
    eps: `float`
        The tolerance to use to determine equality [default=1e-8].
    iterlim: `int`
        The maximum number of iterations if this method has to fall back
        on using Newton's method.  Raises RuntimeError on iteration
        limit [default=1000]
    linesearch: `bool`
        Decides whether or not to use the linesearch (recommended).
        [default=True]
    alpha_min: `float`
        The minimum fractional step to use in the linesearch [default=1e-8].
    diff_mode: :py:enum:`pyomo.core.expr.calculus.derivatives.Modes`
        The mode to use to differentiate the expression.  If
        unspecified, defaults to `Modes.sympy`

    Returns:
    --------
    None

    """
    if not isinstance(constraint, _ConstraintData):
        constraint = Constraint(expr=constraint, name=type(constraint).__name__)
        constraint.construct()
    body = constraint.body
    lower = constraint.lb
    upper = constraint.ub
    if lower != upper:
        raise ValueError(f"Constraint '{constraint}' must be an equality constraint")
    _invalid_types = set(native_complex_types)
    _invalid_types.add(type(None))
    if variable.value is None:
        if variable.lb is None:
            if variable.ub is None:
                variable.set_value(0, skip_validation=True)
            else:
                variable.set_value(min(0, variable.ub), skip_validation=True)
        elif variable.ub is None:
            variable.set_value(max(0, variable.lb), skip_validation=True)
        elif variable.lb <= 0 and variable.ub >= 0:
            variable.set_value(0, skip_validation=True)
        else:
            variable.set_value((variable.lb + variable.ub) / 2.0, skip_validation=True)
    orig_initial_value = variable.value
    x1 = value(variable)
    try:
        residual_1 = value(body)
    except:
        logger.error('Encountered an error evaluating the expression at the initial guess.\n\tPlease provide a different initial guess.')
        raise
    try:
        variable.set_value(x1 - (residual_1 - upper), skip_validation=True)
        residual_2 = value(body, exception=False)
    except OverflowError:
        residual_2 = None
    if residual_2.__class__ not in _invalid_types:
        if abs(residual_2 - upper) < eps:
            variable.set_value(variable.value)
            return
        x2 = value(variable)
        slope = float(residual_1 - residual_2) / (x1 - x2)
        intercept = residual_1 - upper - slope * x1
        if slope:
            variable.set_value(-intercept / slope, skip_validation=True)
            body_val = value(body, exception=False)
            if body_val.__class__ not in _invalid_types and abs(body_val - upper) < eps:
                variable.set_value(variable.value)
                return
    variable.set_value(orig_initial_value, skip_validation=True)
    expr = body - upper
    expr_deriv = None
    if diff_mode in _symbolic_modes:
        try:
            expr_deriv = differentiate(expr, wrt=variable, mode=diff_mode or _default_differentiation_mode)
        except:
            if diff_mode is None:
                logger.debug('Calculating symbolic derivative of expression failed. Reverting to numeric differentiation')
                diff_mode = differentiate.Modes.reverse_numeric
            else:
                raise
        if type(expr_deriv) in native_numeric_types and expr_deriv == 0:
            raise ValueError(f"Variable '{variable}' derivative == 0 in constraint '{constraint}', cannot solve for variable")
    if expr_deriv is None:
        fp0 = differentiate(expr, wrt=variable, mode=diff_mode)
    else:
        fp0 = value(expr_deriv)
    if abs(value(fp0)) < 1e-12:
        raise ValueError(f"Initial value for variable '{variable}' results in a derivative value for constraint '{constraint}' that is very close to zero.\n\tPlease provide a different initial guess.")
    iter_left = iterlim
    fk = residual_1 - upper
    while abs(fk) > eps and iter_left:
        iter_left -= 1
        if not iter_left:
            raise IterationLimitError(f"Iteration limit (%s) reached solving for variable '{variable}' using constraint '{constraint}'; remaining residual = %s" % (iterlim, value(expr)))
        xk = value(variable)
        try:
            fk = value(expr)
            if fk.__class__ in _invalid_types and fk is not None:
                raise ValueError("Complex numbers are not allowed in Newton's method.")
        except:
            logger.error(f"Newton's method encountered an error evaluating the expression for constraint '{constraint}'.\n\tPlease provide a different initial guess or enable the linesearch if you have not.")
            raise
        if expr_deriv is None:
            fpk = differentiate(expr, wrt=variable, mode=diff_mode)
        else:
            fpk = value(expr_deriv)
        if abs(fpk) < 1e-12:
            raise RuntimeError(f"Newton's method encountered a derivative of constraint '{constraint}' with respect to variable '{variable}' that was too close to zero.\n\tPlease provide a different initial guess or enable the linesearch if you have not.")
        pk = -fk / fpk
        alpha = 1.0
        xkp1 = xk + alpha * pk
        variable.set_value(xkp1, skip_validation=True)
        if linesearch:
            c1 = 0.999
            while alpha > alpha_min:
                fkp1 = value(expr, exception=False)
                if fkp1.__class__ in _invalid_types:
                    fkp1 = None
                if fkp1 is not None and fkp1 ** 2 < c1 * fk ** 2:
                    fk = fkp1
                    break
                alpha /= 2.0
                xkp1 = xk + alpha * pk
                variable.set_value(xkp1, skip_validation=True)
            if alpha <= alpha_min:
                residual = value(expr, exception=False)
                if residual.__class__ in _invalid_types:
                    residual = '{function evaluation error}'
                raise IterationLimitError(f"Linesearch iteration limit reached solving for variable '{variable}' using constraint '{constraint}'; remaining residual = {residual}.")
    variable.set_value(variable.value)