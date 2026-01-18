from pythran.analyses import LazynessAnalysis, StrictAliases, YieldPoints
from pythran.analyses import LocalNodeDeclarations, Immediates, RangeValues
from pythran.analyses import Ancestors
from pythran.config import cfg
from pythran.cxxtypes import TypeBuilder, ordered_set
from pythran.intrinsic import UserFunction, Class
from pythran.passmanager import ModuleAnalysis
from pythran.tables import operator_to_lambda, MODULES
from pythran.types.conversion import pytype_to_ctype
from pythran.types.reorder import Reorder
from pythran.utils import attr_to_path, cxxid, isnum, isextslice
from collections import defaultdict
from functools import reduce
import gast as ast
from itertools import islice
from copy import deepcopy
def node_to_id(self, n, depth=()):
    if isinstance(n, ast.Name):
        return (n.id, depth)
    elif isinstance(n, ast.Subscript):
        if isinstance(n.slice, ast.Slice):
            return self.node_to_id(n.value, depth)
        else:
            index = n.slice.value if isnum(n.slice) else None
            return self.node_to_id(n.value, depth + (index,))
    elif isinstance(n, ast.Call):
        for alias in self.strict_aliases[n]:
            if alias is n:
                continue
            try:
                return self.node_to_id(alias, depth)
            except UnboundableRValue:
                continue
    raise UnboundableRValue()