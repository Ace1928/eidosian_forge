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
def configSolvers():
    """
    Configure the path the the solvers on the command line

    Designed to configure the file locations of the solvers from the
    command line after installation
    """
    configlist = [(cplex_dll_path, 'cplexpath', 'CPLEX: '), (coinMP_path, 'coinmppath', 'CoinMP dll (windows only): ')]
    print('Please type the full path including filename and extension \n' + 'for each solver available')
    configdict = {}
    for default, key, msg in configlist:
        value = input(msg + '[' + str(default) + ']')
        if value:
            configdict[key] = value
    setConfigInformation(**configdict)