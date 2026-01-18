from typing import List, Optional, Tuple
import cvxpy.settings as s
import cvxpy.utilities as u
from cvxpy.atoms import sum, trace
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.problems.problem import Problem
def partial_optimize(prob: Problem, opt_vars: Optional[List[Variable]]=None, dont_opt_vars: Optional[List[Variable]]=None, solver=None, **kwargs) -> 'PartialProblem':
    """Partially optimizes the given problem over the specified variables.

    Either opt_vars or dont_opt_vars must be given.
    If both are given, they must contain all the variables in the problem.

    Partial optimize is useful for two-stage optimization and graph implementations.
    For example, we can write

    .. code :: python

        x = Variable(n)
        t = Variable(n)
        abs_x = partial_optimize(Problem(Minimize(sum(t)),
                  [-t <= x, x <= t]), opt_vars=[t])

    to define the entrywise absolute value of x.

    Parameters
    ----------
    prob : Problem
        The problem to partially optimize.
    opt_vars : list, optional
        The variables to optimize over.
    dont_opt_vars : list, optional
        The variables to not optimize over.
    solver : str, optional
        The default solver to use for value and grad.
    kwargs : keywords, optional
        Additional solver specific keyword arguments.

    Returns
    -------
    Expression
        An expression representing the partial optimization.
        Convex for minimization objectives and concave for maximization objectives.
    """
    if opt_vars is None and dont_opt_vars is None:
        raise ValueError('partial_optimize called with neither opt_vars nor dont_opt_vars.')
    elif opt_vars is None:
        ids = [id(var) for var in dont_opt_vars]
        opt_vars = [var for var in prob.variables() if id(var) not in ids]
    elif dont_opt_vars is None:
        ids = [id(var) for var in opt_vars]
        dont_opt_vars = [var for var in prob.variables() if id(var) not in ids]
    elif opt_vars is not None and dont_opt_vars is not None:
        ids = [id(var) for var in opt_vars + dont_opt_vars]
        for var in prob.variables():
            if id(var) not in ids:
                raise ValueError('If opt_vars and new_opt_vars are both specified, they must contain all variables in the problem.')
    id_to_new_var = {id(var): Variable(var.shape, **var.attributes) for var in opt_vars}
    new_obj = prob.objective.tree_copy(id_to_new_var)
    new_constrs = [con.tree_copy(id_to_new_var) for con in prob.constraints]
    new_var_prob = Problem(new_obj, new_constrs)
    return PartialProblem(new_var_prob, opt_vars, dont_opt_vars, solver, **kwargs)