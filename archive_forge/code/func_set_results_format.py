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
def set_results_format(self, format):
    """
        Set the current results format (if it's valid for the current
        problem format).
        """
    if self._problem_format in self._valid_results_formats and format in self._valid_results_formats[self._problem_format]:
        self._results_format = format
    else:
        raise ValueError('%s is not a valid results format for problem format %s with solver plugin %s' % (format, self._problem_format, self))