from .util import subvals
@wrap_nary_f(fun, unary_operator, argnum)
def nary_f(*args, **kwargs):

    @wraps(fun)
    def unary_f(x):
        if isinstance(argnum, int):
            subargs = subvals(args, [(argnum, x)])
        else:
            subargs = subvals(args, zip(argnum, x))
        return fun(*subargs, **kwargs)
    if isinstance(argnum, int):
        x = args[argnum]
    else:
        x = tuple((args[i] for i in argnum))
    return unary_operator(unary_f, x, *nary_op_args, **nary_op_kwargs)