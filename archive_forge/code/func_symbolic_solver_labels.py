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
@symbolic_solver_labels.setter
def symbolic_solver_labels(self, val):
    _raise_ephemeral_error('symbolic_solver_labels')