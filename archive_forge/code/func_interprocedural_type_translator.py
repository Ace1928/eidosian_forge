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
def interprocedural_type_translator(s, n):
    translated_othernode = ast.Name('__fake__', ast.Load(), None, None)
    s.result[translated_othernode] = parametric_type.instanciate(s.current, [s.result[arg] for arg in n.args])
    for p, effective_arg in enumerate(n.args):
        formal_arg = args[p]
        if formal_arg.id == node_id:
            translated_node = effective_arg
            break
    try:
        s.combine(translated_node, op, translated_othernode)
    except NotImplementedError:
        pass
    except UnboundLocalError:
        pass