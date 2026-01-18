import gast as ast
import itertools
import os
from pythran.analyses import GlobalDeclarations
from pythran.errors import PythranInternalError
from pythran.passmanager import ModuleAnalysis
from pythran.types.conversion import PYTYPE_TO_CTYPE_TABLE
from pythran.utils import get_variable
from pythran.typing import List, Set, Dict, NDArray, Tuple, Pointer, Fun
from pythran.graph import DiGraph
def pytype_to_deps_hpp(t):
    """python -> pythonic type hpp filename."""
    if isinstance(t, List):
        return {'list.hpp'}.union(pytype_to_deps_hpp(t.__args__[0]))
    elif isinstance(t, Set):
        return {'set.hpp'}.union(pytype_to_deps_hpp(t.__args__[0]))
    elif isinstance(t, Dict):
        tkey, tvalue = t.__args__
        return {'dict.hpp'}.union(pytype_to_deps_hpp(tkey), pytype_to_deps_hpp(tvalue))
    elif isinstance(t, Tuple):
        return {'tuple.hpp'}.union(*[pytype_to_deps_hpp(elt) for elt in t.__args__])
    elif isinstance(t, NDArray):
        out = {'ndarray.hpp'}
        if t.__args__[1].start == -1:
            out.add('numpy_texpr.hpp')
        return out.union(pytype_to_deps_hpp(t.__args__[0]))
    elif isinstance(t, Pointer):
        return {'pointer.hpp'}.union(pytype_to_deps_hpp(t.__args__[0]))
    elif isinstance(t, Fun):
        return {'cfun.hpp'}.union(*[pytype_to_deps_hpp(a) for a in t.__args__])
    elif t in PYTYPE_TO_CTYPE_TABLE:
        return {'{}.hpp'.format(t.__name__)}
    else:
        raise NotImplementedError('{0}:{1}'.format(type(t), t))