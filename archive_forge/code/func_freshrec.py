import gast as ast
from copy import deepcopy
from numpy import floating, integer, complexfloating
from pythran.tables import MODULES, attributes
import pythran.typing as typing
from pythran.syntax import PythranSyntaxError
from pythran.utils import isnum
def freshrec(tp):
    p = prune(tp)
    if isinstance(p, TypeVariable):
        if is_generic(p, non_generic):
            if p not in mappings:
                mappings[p] = TypeVariable()
            return mappings[p]
        else:
            return p
    elif isinstance(p, dict):
        return p
    elif isinstance(p, Collection):
        return Collection(*[freshrec(x) for x in p.types])
    elif isinstance(p, Scalar):
        return Scalar([freshrec(x) for x in p.types])
    elif isinstance(p, TypeOperator):
        return TypeOperator(p.name, [freshrec(x) for x in p.types])
    elif isinstance(p, MultiType):
        return MultiType([freshrec(x) for x in p.types])
    else:
        assert False, 'missing freshrec case {}'.format(type(p))