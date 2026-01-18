from cvxpy.atoms.elementwise.maximum import maximum
from cvxpy.reductions.eliminate_pwl.canonicalizers.maximum_canon import (
def minimum_canon(expr, args):
    del expr
    tmp = maximum(*[-arg for arg in args])
    canon, constr = maximum_canon(tmp, tmp.args)
    return (-canon, constr)