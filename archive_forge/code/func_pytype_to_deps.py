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
def pytype_to_deps(t):
    """ python -> pythonic type header full path. """
    res = set()
    for hpp_dep in pytype_to_deps_hpp(t):
        res.add(os.path.join('pythonic', 'types', hpp_dep))
        res.add(os.path.join('pythonic', 'include', 'types', hpp_dep))
    return res