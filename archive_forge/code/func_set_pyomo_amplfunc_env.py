import itertools
import logging
import operator
import os
import time
from math import isclose
from pyomo.common.fileutils import find_library
from pyomo.common.gc_manager import PauseGC
from pyomo.opt import ProblemFormat, AbstractProblemWriter, WriterFactory
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import (
from pyomo.core.base import (
import pyomo.core.base.suffix
from pyomo.repn.standard_repn import generate_standard_repn
import pyomo.core.kernel.suffix
from pyomo.core.kernel.block import IBlock
from pyomo.core.kernel.expression import IIdentityExpression
from pyomo.core.kernel.variable import IVariable
def set_pyomo_amplfunc_env(external_libs):
    env_str = ''
    for _lib in external_libs:
        _lib = _lib.strip()
        _abs_lib = find_library(_lib)
        if _abs_lib is not None:
            _lib = _abs_lib
        if ' ' not in _lib or (_lib[0] == '"' and _lib[-1] == '"' and ('"' not in _lib[1:-1])) or (_lib[0] == "'" and _lib[-1] == "'" and ("'" not in _lib[1:-1])):
            pass
        elif '"' not in _lib:
            _lib = '"' + _lib + '"'
        elif "'" not in _lib:
            _lib = "'" + _lib + "'"
        else:
            raise RuntimeError('Cannot pass the AMPL external function library\n\t%s\nto the ASL because the string contains spaces, single quote and\ndouble quote characters.' % (_lib,))
        if env_str:
            env_str += '\n'
        env_str += _lib
    os.environ['PYOMO_AMPLFUNC'] = env_str