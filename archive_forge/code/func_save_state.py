import gast as ast
from collections import defaultdict
from functools import reduce
from pythran.analyses import Aliases, CFG
from pythran.intrinsic import Intrinsic
from pythran.passmanager import ModuleAnalysis
from pythran.interval import Interval, IntervalTuple, UNKNOWN_RANGE
from pythran.tables import MODULES, attributes
def save_state(self):
    return (self.cfg, self.aliases, self.use_omp, self.no_backward, self.no_if_split, self.result.copy())