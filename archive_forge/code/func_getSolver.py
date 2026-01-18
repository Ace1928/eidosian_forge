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
def getSolver(solver, *args, **kwargs):
    """
    Instantiates a solver from its name

    :param str solver: solver name to create
    :param args: additional arguments to the solver
    :param kwargs: additional keyword arguments to the solver
    :return: solver of type :py:class:`LpSolver`
    """
    mapping = {k.name: k for k in _all_solvers}
    try:
        return mapping[solver](*args, **kwargs)
    except KeyError:
        raise PulpSolverError('The solver {} does not exist in PuLP.\nPossible options are: \n{}'.format(solver, mapping.keys()))