import sys
def debugx(expr, pre_msg=''):
    """Print the value of an expression from the caller's frame.

    Takes an expression, evaluates it in the caller's frame and prints both
    the given expression and the resulting value (as well as a debug mark
    indicating the name of the calling function.  The input must be of a form
    suitable for eval().

    An optional message can be passed, which will be prepended to the printed
    expr->value pair."""
    cf = sys._getframe(1)
    print('[DBG:%s] %s%s -> %r' % (cf.f_code.co_name, pre_msg, expr, eval(expr, cf.f_globals, cf.f_locals)))