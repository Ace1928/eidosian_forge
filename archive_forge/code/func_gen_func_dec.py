import re
import sys
import inspect
import operator
import itertools
import collections
from inspect import getfullargspec
def gen_func_dec(func):
    """Decorator turning a function into a generic function"""
    argset = set(getfullargspec(func).args)
    if not set(dispatch_args) <= argset:
        raise NameError('Unknown dispatch arguments %s' % dispatch_str)
    typemap = {}

    def vancestors(*types):
        """
            Get a list of sets of virtual ancestors for the given types
            """
        check(types)
        ras = [[] for _ in range(len(dispatch_args))]
        for types_ in typemap:
            for t, type_, ra in zip(types, types_, ras):
                if issubclass(t, type_) and type_ not in t.__mro__:
                    append(type_, ra)
        return [set(ra) for ra in ras]

    def ancestors(*types):
        """
            Get a list of virtual MROs, one for each type
            """
        check(types)
        lists = []
        for t, vas in zip(types, vancestors(*types)):
            n_vas = len(vas)
            if n_vas > 1:
                raise RuntimeError(f'Ambiguous dispatch for {t}: {vas}')
            elif n_vas == 1:
                va, = vas
                mro = type('t', (t, va), {}).__mro__[1:]
            else:
                mro = t.__mro__
            lists.append(mro[:-1])
        return lists

    def register(*types):
        """
            Decorator to register an implementation for the given types
            """
        check(types)

        def dec(f):
            check(getfullargspec(f).args, operator.lt, ' in ' + f.__name__)
            typemap[types] = f
            return f
        return dec

    def dispatch_info(*types):
        """
            An utility to introspect the dispatch algorithm
            """
        check(types)
        lst = [tuple((a.__name__ for a in anc)) for anc in itertools.product(*ancestors(*types))]
        return lst

    def _dispatch(dispatch_args, *args, **kw):
        types = tuple((type(arg) for arg in dispatch_args))
        try:
            f = typemap[types]
        except KeyError:
            pass
        else:
            return f(*args, **kw)
        combinations = itertools.product(*ancestors(*types))
        next(combinations)
        for types_ in combinations:
            f = typemap.get(types_)
            if f is not None:
                return f(*args, **kw)
        return func(*args, **kw)
    return FunctionMaker.create(func, 'return _f_(%s, %%(shortsignature)s)' % dispatch_str, dict(_f_=_dispatch), register=register, default=func, typemap=typemap, vancestors=vancestors, ancestors=ancestors, dispatch_info=dispatch_info, __wrapped__=func)