from io import StringIO
from pyomo.common.gc_manager import PauseGC
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.visitor import _ToStringVisitor
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter, WriterFactory
from pyomo.repn.util import valid_expr_ctypes_minlp, valid_active_ctypes_minlp, ftoa
import logging
def split_long_line(line):
    """
    GAMS has an 80,000 character limit for lines, so split as many
    times as needed so as to not have illegal lines.
    """
    new_lines = ''
    while len(line) > 80000:
        i = 80000
        while line[i] != ' ':
            if i < 0:
                raise RuntimeError('Found an 80,000+ character string with no spaces')
            i -= 1
        new_lines += line[:i] + '\n'
        line = line[i:]
    new_lines += line
    return new_lines