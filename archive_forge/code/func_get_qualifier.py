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
def get_qualifier(self, node):
    lazy_res = self.lazyness_analysis[node.id]
    return self.builder.Lazy if lazy_res <= self.max_recompute else self.builder.Assignable