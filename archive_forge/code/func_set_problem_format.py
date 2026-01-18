import os
import re
import time
import logging
import subprocess
from pyomo.common import Executable
from pyomo.common.errors import ApplicationError
from pyomo.common.collections import Bunch
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.kernel.block import IBlock
from pyomo.core import Var
from pyomo.opt.base import ProblemFormat, ResultsFormat, OptSolver
from pyomo.opt.base.solvers import _extract_version, SolverFactory
from pyomo.opt.results import (
from pyomo.opt.solver import SystemCallSolver
from pyomo.solvers.mockmip import MockMIP
def set_problem_format(self, format):
    super(CBCSHELL, self).set_problem_format(format)
    if self._problem_format == ProblemFormat.cpxlp:
        self._capabilities.sos1 = False
        self._capabilities.sos2 = False
    else:
        self._capabilities.sos1 = True
        self._capabilities.sos2 = True
    if self._problem_format == ProblemFormat.nl:
        if self._compiled_with_asl():
            _ver = self.version()
            if not _ver or _ver[:3] < (2, 7, 0):
                if _ver is None:
                    _ver_str = '<unknown>'
                else:
                    _ver_str = '.'.join((str(i) for i in _ver))
                logger.warning(f'found CBC version {_ver_str} < 2.7; ASL support disabled (falling back on LP interface).')
                logger.warning('Upgrade CBC to activate ASL support in this plugin')
                self.set_problem_format(ProblemFormat.cpxlp)
        else:
            logger.warning('CBC solver is not compiled with ASL interface (falling back on LP interface).')
            self.set_problem_format(ProblemFormat.cpxlp)