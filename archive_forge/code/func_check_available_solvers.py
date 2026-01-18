import re
import sys
import time
import logging
import shlex
from pyomo.common import Factory
from pyomo.common.errors import ApplicationError
from pyomo.common.collections import Bunch
from pyomo.opt.base.convert import convert_problem
from pyomo.opt.base.formats import ResultsFormat
import pyomo.opt.base.results
def check_available_solvers(*args):
    from pyomo.solvers.plugins.solvers.GUROBI import GUROBISHELL
    from pyomo.solvers.plugins.solvers.BARON import BARONSHELL
    from pyomo.solvers.plugins.solvers.mosek_direct import MOSEKDirect
    logging.disable(logging.WARNING)
    ans = []
    for arg in args:
        if not isinstance(arg, tuple):
            name = arg
            arg = (arg,)
        else:
            name = arg[0]
        opt = SolverFactory(*arg)
        if opt is None or isinstance(opt, UnknownSolver):
            continue
        if not opt.available(exception_flag=False):
            continue
        if hasattr(opt, 'executable') and opt.executable() is None:
            continue
        if not opt.license_is_valid():
            continue
        ans.append(name)
    logging.disable(logging.NOTSET)
    return ans