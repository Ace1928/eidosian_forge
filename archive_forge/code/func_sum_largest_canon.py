from cvxpy.atoms.affine.sum import sum
from cvxpy.expressions.variable import Variable
def sum_largest_canon(expr, args):
    x = args[0]
    k = expr.k
    t = Variable(x.shape)
    q = Variable()
    obj = sum(t) + k * q
    constraints = [x <= t + q, t >= 0]
    return (obj, constraints)