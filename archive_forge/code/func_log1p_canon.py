from cvxpy.reductions.dcp2cone.canonicalizers.log_canon import log_canon
def log1p_canon(expr, args):
    return log_canon(expr, [args[0] + 1])