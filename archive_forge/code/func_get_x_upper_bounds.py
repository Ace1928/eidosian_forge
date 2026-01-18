from pyomo.common.fileutils import find_library
from pyomo.common.dependencies import numpy as np
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
import ctypes
import logging
import os
def get_x_upper_bounds(self, invec):
    self.ASLib.EXTERNAL_AmplInterface_x_upper_bounds(self._obj, invec, len(invec))