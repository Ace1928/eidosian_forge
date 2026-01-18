from pythran.analyses.global_declarations import GlobalDeclarations
from pythran.intrinsic import Intrinsic, Class, UnboundValue
from pythran.passmanager import ModuleAnalysis
from pythran.tables import functions, methods, MODULES
from pythran.unparse import Unparser
from pythran.conversion import demangle
import pythran.metadata as md
from pythran.utils import isnum
from pythran.syntax import PythranSyntaxError
import gast as ast
from copy import deepcopy
from itertools import product
import io
def pp(n):
    output = io.StringIO()
    if isinstance(n, ContainerOf):
        if n.index == n.index:
            output.write('[{}]='.format(n.index))
        containees = sorted(map(pp, n.containees))
        output.write(', '.join(map('|{}|'.format, containees)))
    else:
        Unparser(n, output)
    return output.getvalue().strip()