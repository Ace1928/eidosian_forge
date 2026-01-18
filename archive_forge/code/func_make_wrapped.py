from functools import wraps
from .sympify import SympifyError, sympify
def make_wrapped(self, cls):
    func = self.func
    parameters, retval = self.args
    [(parameter, expectedcls)] = parameters
    if expectedcls == cls.__name__:
        expectedcls = cls
    nargs = func.__code__.co_argcount
    if nargs != 2:
        raise RuntimeError('sympify_return can only be used with 2 argument functions')
    if func.__code__.co_varnames[1] != parameter:
        raise RuntimeError('parameter name mismatch "%s" in %s' % (parameter, func.__name__))

    @wraps(func)
    def _func(self, other):
        if not hasattr(other, '_op_priority'):
            try:
                other = sympify(other, strict=True)
            except SympifyError:
                return retval
        if not isinstance(other, expectedcls):
            return retval
        return func(self, other)
    return _func