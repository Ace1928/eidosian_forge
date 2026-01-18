from ..libmp.backend import xrange
from .calculus import defun
def iterable_to_function(gen):
    gen = iter(gen)
    data = []

    def f(k):
        for i in xrange(len(data), k + 1):
            data.append(next(gen))
        return data[k]
    return f