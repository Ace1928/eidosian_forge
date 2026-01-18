from .base_linear_solver_interface import IPLinearSolverInterface
from pyomo.contrib.pynumero.linalg.base import LinearSolverStatus, LinearSolverResults
from pyomo.common.dependencies import attempt_import
from collections import OrderedDict
from typing import Union, Optional, Tuple
from pyomo.contrib.pynumero.sparse import BlockVector
import numpy as np
from pyomo.contrib.pynumero.linalg.mumps_interface import (
def log_header(self, include_error=True, extra_fields=None):
    if extra_fields is None:
        extra_fields = list()
    header_fields = []
    header_fields.append('Status')
    header_fields.append('n_null')
    header_fields.append('n_neg')
    if include_error:
        header_fields.extend(self.get_error_info().keys())
    header_fields.extend(extra_fields)
    header_string = '{0:<10}'
    header_string += '{1:<10}'
    header_string += '{2:<10}'
    for i in range(4, len(header_fields)):
        header_string += '{' + str(i) + ':<15}'
    self.logger.info(header_string.format(*header_fields))