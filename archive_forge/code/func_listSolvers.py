from .coin_api import *
from .cplex_api import *
from .gurobi_api import *
from .glpk_api import *
from .choco_api import *
from .mipcl_api import *
from .mosek_api import *
from .scip_api import *
from .xpress_api import *
from .highs_api import *
from .copt_api import *
from .core import *
import json
def listSolvers(onlyAvailable=False):
    """
    List the names of all the existing solvers in PuLP

    :param bool onlyAvailable: if True, only show the available solvers
    :return: list of solver names
    :rtype: list
    """
    result = []
    for s in _all_solvers:
        solver = s()
        if not onlyAvailable or solver.available():
            result.append(solver.name)
        del solver
    return result